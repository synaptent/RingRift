# Runbook: Training Loop Stalled

**Severity**: High
**Expected Resolution Time**: 15-30 minutes
**Last Updated**: December 28, 2025

---

## Symptoms

- Training jobs not progressing (no new checkpoint saves)
- `TRAINING_STARTED` events emitted but no `TRAINING_COMPLETED`
- Data pipeline shows "waiting_for_training" status indefinitely
- GPU utilization drops to 0% on training nodes

---

## Quick Diagnosis

```bash
# Check training jobs in P2P
curl -s http://localhost:8770/status | python3 -c '
import sys, json
d = json.load(sys.stdin)
for job in d.get("jobs", []):
    if "training" in job.get("type", ""):
        print(f"{job[\"id\"]}: {job[\"status\"]} - {job.get(\"node_id\")}")
'

# Check for stuck training processes
ssh training-node 'ps aux | grep -E "train\.py|training" | grep -v grep'

# Check GPU memory usage
ssh training-node 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'

# Check training logs
ssh training-node 'tail -50 ~/ringrift/ai-service/logs/training.log 2>/dev/null || echo "No training log"'
```

---

## Common Causes & Fixes

### 1. GPU Out of Memory (OOM)

**Symptoms**: Training starts but crashes silently
**Check**:

```bash
ssh training-node 'dmesg | tail -20 | grep -i oom'
ssh training-node 'cat ~/ringrift/ai-service/logs/training.log | grep -i "cuda out of memory"'
```

**Fix**: Reduce batch size

```bash
# Restart with smaller batch
ssh training-node 'cd ~/ringrift/ai-service && python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --batch-size 256 \  # Reduced from 512
  --data-path data/training/hex8_2p.npz'
```

### 2. Training Lock Not Released

**Symptoms**: Previous training crashed leaving lock held
**Check**:

```bash
# Check for stale lock
python -c "
from app.coordination.distributed_lock import training_lock
lock = training_lock()
print(f'Locked: {lock.is_locked()}')
print(f'Lock holder: {lock.get_holder()}')
"
```

**Fix**: Release stale lock

```bash
python -c "
from app.coordination.distributed_lock import training_lock
lock = training_lock()
if lock.is_locked():
    print(f'Releasing lock held by: {lock.get_holder()}')
    lock.force_release()
    print('Lock released')
"
```

### 3. Data Pipeline Not Ready

**Symptoms**: Insufficient training data
**Check**:

```bash
# Check game counts
python -c "
from app.utils.game_discovery import GameDiscovery
d = GameDiscovery()
for db in d.find_databases_for_config('hex8', 2):
    print(f'{db.path}: {db.game_count} games')
"

# Check NPZ file
python -c "
import numpy as np
data = np.load('data/training/hex8_2p.npz')
print(f'Samples: {len(data[\"features\"])}')
"
```

**Fix**: Export fresh training data

```bash
python scripts/export_replay_dataset.py \
  --use-discovery \
  --board-type hex8 --num-players 2 \
  --output data/training/hex8_2p.npz
```

### 4. TrainingCoordinator Cooldown Active

**Symptoms**: Training recently completed, cooldown preventing new run
**Check**:

```bash
python -c "
from scripts.p2p.managers import TrainingCoordinator
# Check cooldown status in state manager
print('Cooldown check requires P2P orchestrator access')
"

# Check recent training timestamps
curl -s http://localhost:8770/status | python3 -c '
import sys, json
d = json.load(sys.stdin)
print(f"Last training: {d.get(\"last_training_time\", \"unknown\")}")
'
```

**Fix**: Wait for cooldown (5 minutes) or manually dispatch

```bash
# Force dispatch training job
curl -X POST http://localhost:8770/dispatch_training \
  -H "Content-Type: application/json" \
  -d '{"config": "hex8_2p", "force": true}'
```

### 5. Event System Not Wired

**Symptoms**: TRAINING_COMPLETED events never emitted
**Check**:

```bash
python -c "
from app.coordination.event_router import get_router
router = get_router()
print(f'TRAINING_COMPLETED subscribers: {len(router.get_subscribers(\"TRAINING_COMPLETED\"))}')
"
```

**Fix**: Restart coordination bootstrap

```bash
python scripts/launch_daemons.py --restart EVENT_ROUTER FEEDBACK_LOOP DATA_PIPELINE
```

### 6. Model Checkpoint Corruption

**Symptoms**: Training loads corrupted checkpoint and fails
**Check**:

```bash
# Verify checkpoint loadability
python -c "
import torch
from app.utils.torch_utils import safe_load_checkpoint
try:
    cp = safe_load_checkpoint('models/checkpoint_hex8_2p.pth')
    print(f'Checkpoint OK: {cp.keys()}')
except Exception as e:
    print(f'Checkpoint corrupt: {e}')
"
```

**Fix**: Delete corrupted checkpoint and restart from last good

```bash
# Backup corrupt checkpoint
mv models/checkpoint_hex8_2p.pth models/checkpoint_hex8_2p.pth.corrupt

# Copy canonical model as starting point
cp models/canonical_hex8_2p.pth models/checkpoint_hex8_2p.pth
```

---

## Full Recovery Procedure

If quick fixes don't work:

### Step 1: Stop All Training

```bash
# Kill training processes on all nodes
for node in nebius-h100-3 runpod-h100 lambda-gh200-b-new; do
  ssh $node 'pkill -f "train.py" || true'
done
```

### Step 2: Clear Training State

```bash
# Release distributed lock
python -c "
from app.coordination.distributed_lock import training_lock
lock = training_lock()
lock.force_release()
print('Lock released')
"

# Clear P2P training job records
curl -X POST http://localhost:8770/clear_training_state \
  -H "Content-Type: application/json"
```

### Step 3: Verify Data Ready

```bash
# Export fresh NPZ
python scripts/export_replay_dataset.py \
  --use-discovery \
  --board-type hex8 --num-players 2 \
  --output data/training/hex8_2p.npz

# Sync to training node
rsync -avz data/training/hex8_2p.npz \
  ubuntu@nebius-h100-3:~/ringrift/ai-service/data/training/
```

### Step 4: Restart Training

```bash
# Via P2P (recommended)
curl -X POST http://localhost:8770/dispatch_training \
  -H "Content-Type: application/json" \
  -d '{"config": "hex8_2p"}'

# Or manually
ssh nebius-h100-3 'cd ~/ringrift/ai-service && nohup python -m app.training.train \
  --board-type hex8 --num-players 2 \
  --data-path data/training/hex8_2p.npz \
  --batch-size 512 --epochs 50 \
  > logs/training.log 2>&1 &'
```

### Step 5: Monitor Progress

```bash
# Watch training log
ssh nebius-h100-3 'tail -f ~/ringrift/ai-service/logs/training.log'

# Check GPU utilization
watch -n 5 'ssh nebius-h100-3 "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"'
```

---

## Prevention

1. **Enable health monitoring**: Ensure `FEEDBACK_LOOP` daemon is running
2. **Set appropriate batch sizes**: Use `--batch-size 256` for 8GB VRAM, `--batch-size 512` for 16GB+
3. **Monitor disk space**: Training needs ~10GB free for checkpoints
4. **Regular checkpoint cleanup**: Keep only last 5 checkpoints per config

---

## Related Events

| Event                | Emitter             | Subscribers                |
| -------------------- | ------------------- | -------------------------- |
| `TRAINING_STARTED`   | TrainingCoordinator | SyncRouter, IdleShutdown   |
| `TRAINING_COMPLETED` | TrainingCoordinator | FeedbackLoop, DataPipeline |
| `TRAINING_FAILED`    | TrainingCoordinator | AlertManager               |

---

## Related Runbooks

- [GPU_OOM_DEBUG.md](GPU_OOM_DEBUG.md) - GPU memory issues
- [DAEMON_FAILURE_RECOVERY.md](DAEMON_FAILURE_RECOVERY.md) - Daemon issues
- [CLUSTER_GPU_STUCK.md](CLUSTER_GPU_STUCK.md) - Stuck GPU processes
