# Debugging Guide

Common issues and solutions for the RingRift AI Service.

## Table of Contents

1. [P2P Cluster Issues](#p2p-cluster-issues)
2. [Data Sync Issues](#data-sync-issues)
3. [Training Issues](#training-issues)
4. [Parity and Replay Issues](#parity-and-replay-issues)
5. [GPU and Resource Issues](#gpu-and-resource-issues)
6. [Daemon Issues](#daemon-issues)
7. [Event System Issues](#event-system-issues)

---

## P2P Cluster Issues

### P2P Orchestrator Won't Start

**Symptom**: `scripts/p2p_orchestrator.py` fails to start or crashes immediately.

**Common Causes**:

1. **Port 8770 already in use**

   ```bash
   # Check what's using the port
   lsof -i :8770

   # Kill existing process
   pkill -f p2p_orchestrator
   ```

2. **Missing dependencies**

   ```bash
   # Check required modules
   python3 -c "import aiohttp, psutil, yaml; print('OK')"

   # Install if missing
   pip install aiohttp psutil pyyaml
   ```

3. **/dev/shm not available (macOS)**
   - The P2P orchestrator will automatically fall back to disk storage
   - Check logs for: "Falling back to disk storage"

### Leader Election Stuck

**Symptom**: Cluster shows no leader, all nodes are "follower".

**Debug Steps**:

```bash
# Check cluster status
curl -s http://localhost:8770/status | python3 -c '
import sys,json
d = json.load(sys.stdin)
print(f"Leader: {d.get(\"leader_id\")}")
print(f"Alive peers: {d.get(\"alive_peers\")}")
print(f"My role: {d.get(\"role\")}")
'

# Check voter quorum (need 3 of 5 voters alive)
curl -s http://localhost:8770/status | jq '.voter_status'
```

**Common Fixes**:

- Ensure at least 3 of 5 voter nodes are reachable
- Check Tailscale connectivity: `tailscale status`
- Restart P2P on problematic nodes

### Node Not Joining Cluster

**Symptom**: Node shows as "offline" in cluster status.

**Debug Steps**:

```bash
# Check if node can reach other nodes
for ip in 100.x.x.x 100.y.y.y; do
  curl -s --connect-timeout 3 "http://$ip:8770/health" && echo " - $ip OK"
done

# Check firewall
sudo iptables -L -n | grep 8770
```

**Common Fixes**:

- Add node to `config/distributed_hosts.yaml`
- Ensure Tailscale is connected
- Check SSH connectivity: `ssh -o ConnectTimeout=5 user@host echo OK`

---

## Data Sync Issues

### SCP/Rsync "Connection Reset by Peer"

**Symptom**: File transfers fail with "Connection reset by peer" or "Broken pipe".

**Solution**: Use base64 transfer fallback:

```bash
# Manual base64 transfer
cat local_file.npz | base64 | ssh user@host 'base64 -d > remote_file.npz'

# Or use the robust_push function
python3 -c "
from scripts.lib.transfer import robust_push, TransferConfig
result = robust_push('file.npz', 'host', 22, '/path/file.npz', TransferConfig())
print(f'Success: {result.success}')
"
```

### Sync Daemon Not Running

**Symptom**: Data not being synced, stale databases on nodes.

**Debug Steps**:

```bash
# Check daemon status
python3 -c "
from app.coordination.daemon_manager import DaemonManager
from app.coordination.daemon_types import DaemonType
mgr = DaemonManager.get_instance()
info = mgr.get_daemon_info(DaemonType.AUTO_SYNC)
print(f'State: {info.state if info else \"Not registered\"}')
"

# Check recent sync logs
grep -r "DATA_SYNC" logs/*.log | tail -20
```

### Database Locked Errors

**Symptom**: "database is locked" errors during sync.

**Common Fixes**:

1. Use context managers for all database operations
2. Set appropriate timeout: `sqlite3.connect(db, timeout=30.0)`
3. Check for zombie processes holding locks:
   ```bash
   fuser -v *.db  # Linux
   lsof *.db      # macOS
   ```

---

## Training Issues

### Training Data Stale

**Symptom**: Training uses old data, Elo not improving.

**Debug Steps**:

```bash
# Check data freshness
python3 -c "
from app.coordination.training_freshness import TrainingFreshnessChecker
checker = TrainingFreshnessChecker()
status = checker.check_freshness('hex8', 2)
print(f'Fresh: {status.is_fresh}, Age: {status.age_hours:.1f}h')
"

# Force data sync before training
python -m app.training.train --check-data-freshness --max-data-age-hours 1.0
```

### Model Distribution Failed

**Symptom**: After training, other nodes don't have the new model.

**Debug Steps**:

```bash
# Check MODEL_PROMOTED event was emitted
grep "MODEL_PROMOTED" logs/*.log | tail -5

# Check distribution daemon status
python3 -c "
from app.coordination.daemon_manager import DaemonManager
from app.coordination.daemon_types import DaemonType
mgr = DaemonManager.get_instance()
info = mgr.get_daemon_info(DaemonType.MODEL_DISTRIBUTION)
print(f'State: {info.state if info else \"Not registered\"}')
"

# Manual distribution
python scripts/sync_models.py --distribute
```

### Out of GPU Memory

**Symptom**: CUDA out of memory errors during training.

**Common Fixes**:

1. Reduce batch size: `--batch-size 256` instead of 512
2. Enable gradient checkpointing: `--gradient-checkpointing`
3. Use mixed precision: `--fp16`
4. Check for memory leaks:
   ```bash
   nvidia-smi -l 1  # Watch memory usage
   ```

---

## Parity and Replay Issues

### Parity Gate Stuck on "pending_gate"

**Symptom**: All games show `pending_gate` status, can't export training data.

**Root Cause**: Cluster nodes don't have `npx` (Node.js) installed.

**Workaround**:

```bash
# Allow pending gate databases
export RINGRIFT_ALLOW_PENDING_GATE=true

# Then run export normally
python scripts/export_replay_dataset.py --use-discovery ...

# Or run parity locally (where npx is available) before syncing
python scripts/check_ts_python_replay_parity.py --db data/games/my_games.db
```

### Replay Mismatch at Move N

**Symptom**: "Replay diverged at move X" errors.

**Debug Steps**:

```bash
# Dump state at the problematic move
RINGRIFT_TS_REPLAY_DUMP_STATE_AT_K=X \
RINGRIFT_TS_REPLAY_DUMP_DIR=/tmp \
TS_NODE_PROJECT=tsconfig.server.json \
npx ts-node scripts/selfplay-db-ts-replay.ts --db mydb.db --game GAME_ID

# Compare Python and TypeScript states
python3 -c "
import json
with open('/tmp/state_k_X.json') as f:
    ts_state = json.load(f)
# Compare with Python engine state at same move
"
```

**Common Causes**:

- Chain capture FSM mismatch
- Territory scoring differences (use `--legacy-scoring` flag)
- Move validation differences

### Move Data Corrupted (to=None)

**Symptom**: All moves have `to=None`, 100% replay failure.

**Root Cause**: Phase extracted from post-move state instead of pre-move state.

**Fix Applied**: Commit `d840f4c4` - use explicit pre-move state capture.

**Recovery**: Regenerate affected database:

```bash
python scripts/selfplay.py --board hex8 --num-players 4 --engine gumbel --num-games 1000
```

---

## GPU and Resource Issues

### GPU Idle Detection Killing Processes

**Symptom**: Selfplay processes killed unexpectedly.

**Debug**:

```bash
# Check idle daemon status
grep "IDLE_RESOURCE" logs/*.log | tail -10

# Adjust thresholds
export RINGRIFT_GPU_IDLE_THRESHOLD=1200  # 20 minutes instead of 10
export RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD=256  # Higher limit
```

### MPS (Apple Silicon) Slow

**Symptom**: GPU selfplay slower than CPU on Mac.

**Root Cause**: MPS has high kernel launch overhead.

**Solution**: Use CPU on Mac:

```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU
python scripts/selfplay.py ...
```

### CUDA Device Mismatch

**Symptom**: "CUDA error: device-side assert triggered" or tensor device mismatches.

**Common Fixes**:

```python
# Ensure all tensors on same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = inputs.to(device)

# Check current device
print(f"Model device: {next(model.parameters()).device}")
```

---

## Daemon Issues

### Daemon Crash Loop

**Symptom**: Daemon starts, crashes, restarts repeatedly.

**Debug Steps**:

```bash
# Check daemon health
python3 -c "
from app.coordination.daemon_manager import DaemonManager
from app.coordination.daemon_types import DaemonType
mgr = DaemonManager.get_instance()
info = mgr.get_daemon_info(DaemonType.YOUR_DAEMON)
if info:
    print(f'State: {info.state}')
    print(f'Restarts: {info.restart_count}')
    print(f'Last error: {info.last_error}')
"

# Check logs for the specific daemon
grep "YOUR_DAEMON" logs/*.log | tail -50
```

### Health Check Loop Not Running

**Symptom**: Crashed daemons not auto-restarting.

**Root Cause**: Health loop only started in `start_all()`, not individual `start()`.

**Fix Applied**: Commit with `_ensure_health_loop_running()` helper.

**Manual Restart**:

```python
from app.coordination.daemon_manager import DaemonManager
from app.coordination.daemon_types import DaemonType

mgr = DaemonManager.get_instance()
await mgr.restart(DaemonType.YOUR_DAEMON)
```

### Daemon Dependencies Not Met

**Symptom**: "Dependency X not running" errors.

**Debug**:

```bash
# Check dependency tree
python3 -c "
from app.coordination.daemon_manager import DaemonManager
from app.coordination.daemon_types import DaemonType, DAEMON_STARTUP_ORDER
for dt in DAEMON_STARTUP_ORDER[:10]:
    print(dt.name)
"

# Start dependencies first
python3 -c "
import asyncio
from app.coordination.daemon_manager import DaemonManager
from app.coordination.daemon_types import DaemonType

async def start():
    mgr = DaemonManager.get_instance()
    await mgr.start(DaemonType.EVENT_ROUTER)  # Usually the first dependency

asyncio.run(start())
"
```

---

## Event System Issues

### Event Not Reaching Handler

**Symptom**: Event emitted but handler never called.

**Debug Steps**:

```python
# Check if handler is subscribed
from app.coordination.event_router import get_event_bus, DataEventType

bus = get_event_bus()
print(f"Subscribers for TRAINING_COMPLETED: {bus._subscribers.get(DataEventType.TRAINING_COMPLETED, [])}")

# Add debug logging
import logging
logging.getLogger("app.coordination.event_router").setLevel(logging.DEBUG)
```

**Common Causes**:

1. Handler not subscribed (check `subscribe_to_events()` was called)
2. Event type mismatch (string vs enum)
3. Handler is async but not awaited
4. Exception in handler being silently caught

### Event Deduplication Blocking Events

**Symptom**: Same event only processed once even when expected multiple times.

**Root Cause**: SHA256 content-based deduplication.

**Fix**: Add unique field to payload:

```python
payload = {
    "config_key": "hex8_2p",
    "timestamp": time.time(),  # Makes each event unique
    "event_id": str(uuid.uuid4()),  # Or use explicit ID
}
```

### Sync Events Not Triggering Pipeline

**Symptom**: DATA_SYNC_COMPLETED emitted but export doesn't start.

**Debug**:

```bash
# Check event type value matches
python3 -c "
from app.coordination.event_router import DataEventType
print(f'Expected: {DataEventType.DATA_SYNC_COMPLETED.value}')
"

# Check DataPipelineOrchestrator subscription
grep "DATA_SYNC_COMPLETED" app/coordination/data_pipeline_orchestrator.py
```

**Fix Applied**: Commit with `_get_event_type_value()` helper in sync_planner.py.

---

## Quick Diagnostic Commands

```bash
# Overall cluster health
curl -s http://localhost:8770/status | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"Healthy: {d.get(\"is_healthy\")}, Leader: {d.get(\"leader_id\")}, Alive: {d.get(\"alive_peers\")}")'

# Check all daemon states
python3 -c "
from app.coordination.daemon_manager import DaemonManager
mgr = DaemonManager.get_instance()
for name, info in mgr._daemons.items():
    print(f'{name}: {info.state.name}')
"

# Recent errors in logs
grep -i "error\|exception\|failed" logs/*.log | tail -30

# GPU status
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

# Disk space on data directories
du -sh data/games data/training models

# Database integrity
sqlite3 data/games/canonical_hex8_2p.db "PRAGMA integrity_check"
```

---

## Getting Help

If you can't resolve an issue:

1. Check existing issues: https://github.com/anthropics/claude-code/issues
2. Include in your report:
   - Error message and stack trace
   - Output of relevant diagnostic commands above
   - Node configuration (GPU type, provider, OS)
   - Recent changes that might have triggered the issue
