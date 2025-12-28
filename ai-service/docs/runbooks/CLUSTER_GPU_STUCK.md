# GPU Node Stuck in Job

Runbook for diagnosing and resolving GPU nodes that appear stuck.

## Symptoms

- Node reports `selfplay_jobs > 0` in P2P `/status`
- GPU utilization shows 0% or < 5%
- No games being generated (database not growing)
- Node appears "busy" but not producing output

## Quick Commands

```bash
# Check P2P status for a specific node
curl -s http://leader:8770/status | jq '.peers["NODE_ID"]'

# Check all nodes with jobs but low GPU
curl -s http://leader:8770/status | jq '
  .peers | to_entries[] |
  select(.value.selfplay_jobs > 0 and .value.gpu_percent < 5) |
  {node: .key, jobs: .value.selfplay_jobs, gpu: .value.gpu_percent}
'

# Auto-cleanup zombies (Dec 2025)
cd ai-service && python scripts/auto_start_idle_selfplay.py --cleanup-only
```

## Diagnosis

### 1. Check P2P Status

```bash
curl -s http://localhost:8770/status | python3 -c '
import sys, json
d = json.load(sys.stdin)
for node_id, peer in d.get("peers", {}).items():
    jobs = peer.get("selfplay_jobs", 0)
    gpu = peer.get("gpu_percent", 0)
    if jobs > 0 and gpu < 5:
        print(f"ZOMBIE: {node_id} - jobs={jobs}, gpu={gpu}%")
'
```

### 2. SSH to Node and Check Processes

```bash
# Get SSH details from distributed_hosts.yaml
ssh -p PORT root@HOST

# Check for Python processes
ps aux | grep -E "python.*selfplay|python.*train"

# Check GPU usage
nvidia-smi

# Check for stuck processes (running > 2 hours with no GPU activity)
ps aux | grep python | awk '$10 > "2:00:00" {print $0}'
```

### 3. Check Database Growth

```bash
# On the node
ls -la data/games/*.db
# Check if file sizes are growing

# Check game count
sqlite3 data/games/selfplay.db "SELECT COUNT(*) FROM games"
```

## Resolution

### Option 1: Automated Cleanup (Recommended)

```bash
# From coordinator node (has SSH access to cluster)
cd ai-service

# Clean up all zombie nodes
python scripts/auto_start_idle_selfplay.py --cleanup-only

# Or clean up and restart selfplay
python scripts/auto_start_idle_selfplay.py --cleanup-zombies
```

### Option 2: Manual Process Cleanup

```bash
# SSH to the stuck node
ssh -p PORT root@HOST

# Kill all selfplay processes
pkill -f 'python.*selfplay'

# Verify GPU is now idle
nvidia-smi
```

### Option 3: Reset Job Count on Leader

If processes are killed but P2P still shows inflated job count:

```bash
# Reset job count for specific node
curl -X POST http://leader:8770/admin/reset_node_jobs \
  -H "Content-Type: application/json" \
  -d '{"node_id": "NODE_ID_HERE"}'
```

## Prevention

### Zombie Detection (Dec 2025)

The `IdleDetectionLoop` now automatically detects zombie nodes:

- Nodes with `selfplay_jobs > 0` AND `gpu_percent < 5%` for > 10 minutes
- Triggers `on_zombie_detected` callback when threshold exceeded
- Check zombie status:

```bash
curl -s http://localhost:8770/status | jq '.idle_detection_stats.zombie_nodes'
```

### Automatic Cleanup in Daemon Mode

```bash
# Run auto-start with zombie cleanup enabled
python scripts/auto_start_idle_selfplay.py --daemon --cleanup-zombies
```

## Related

- [DAEMON_FAILURE_RECOVERY.md](./DAEMON_FAILURE_RECOVERY.md) - Daemon restart procedures
- [P2P_LEADER_FAILOVER.md](./P2P_LEADER_FAILOVER.md) - Leader election issues
- [GAME_HEALTH.md](./GAME_HEALTH.md) - Game database health checks

## History

- Dec 2025: Added zombie detection to IdleDetectionLoop
- Dec 2025: Added `/admin/reset_node_jobs` endpoint
- Dec 2025: Added `--cleanup-zombies` flag to auto_start_idle_selfplay.py
