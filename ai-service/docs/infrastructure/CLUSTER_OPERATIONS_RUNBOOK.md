# RingRift Cluster Operations Runbook

Quick reference for daily cluster operations and troubleshooting.

## Cluster Overview

Configure your cluster in `config/cluster_hosts.yaml`. Example structure:

| Host Type    | Role               | GPU Example | Notes            |
| ------------ | ------------------ | ----------- | ---------------- |
| gpu-primary  | Primary, Training  | H100/GH200  | Main coordinator |
| gpu-worker-N | Workers            | Any GPU     | Selfplay workers |
| gpu-training | Training + Workers | H100+       | High-throughput  |

**Note:** Update `config/cluster_hosts.yaml` with your actual host definitions.

## Quick Status Check

### 1. Check All Nodes

```bash
# From local machine
./ai-service/scripts/cluster_status.sh

# Or manually check each host
for host in gpu-primary gpu-worker-{1..4} gpu-training; do
    echo -n "$host: "
    ssh -o ConnectTimeout=3 $host 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader' 2>/dev/null || echo "offline"
done
```

### 2. Check Running Services

```bash
ssh gpu-primary 'systemctl status ringrift-*'
```

### 3. Check Worker Count

```bash
ssh gpu-primary 'ps aux | grep -E "(selfplay|train)" | grep -v grep | wc -l'
```

## Common Operations

### Update All Cluster Nodes (NEW - Dec 2025)

Update all nodes to the latest code in parallel:

```bash
# From coordinator (mac-studio or local-mac)
cd ~/Development/RingRift/ai-service

# Update all nodes to latest git commit
python scripts/update_all_nodes.py

# Update with P2P orchestrator restart
python scripts/update_all_nodes.py --restart-p2p

# Preview changes without applying
python scripts/update_all_nodes.py --dry-run

# Update to specific commit
python scripts/update_all_nodes.py --commit abc1234

# Limit parallel updates (default: 10)
python scripts/update_all_nodes.py --max-parallel 5
```

**Features:**

- Updates all nodes in `config/distributed_hosts.yaml`
- Skips coordinator nodes and nodes with status != ready
- Auto-detects paths by provider (RunPod, Vast, Nebius, etc.)
- Git stash before pull, commit verification after
- Optional P2P orchestrator restart
- Detailed summary report

**Example output:**

```
✅ Successfully updated: 45 nodes
⏭️  Skipped: 4 nodes (coordinator, retired)
❌ Failed: 3 nodes (connection timeout)
📊 Total: 52 nodes
```

### Start/Stop GPU Selfplay Workers

Supported board types: `square8`, `hex8`, `square19`, `hexagonal`

```bash
# Start on a node (square8)
ssh gpu-worker-1 'cd ~/ringrift/ai-service && \
  nohup venv/bin/python scripts/run_gpu_selfplay.py \
    --board square8 --num-players 2 --num-games 1000 \
    --output-dir data/selfplay/gpu_square8_2p > /tmp/gpu_selfplay.log 2>&1 &'

# Start hex8 selfplay (radius-4 hexagonal, 61 cells)
ssh gpu-worker-1 'cd ~/ringrift/ai-service && \
  nohup venv/bin/python scripts/run_gpu_selfplay.py \
    --board hex8 --num-players 2 --num-games 1000 \
    --output-dir data/selfplay/gpu_hex8_2p > /tmp/gpu_selfplay_hex8.log 2>&1 &'

# Stop workers on a node
ssh gpu-worker-1 'pkill -f run_gpu_selfplay'
```

### Trigger Training

```bash
# Standard NNUE training
ssh gpu-primary 'cd ~/ringrift/ai-service && \
  venv/bin/python scripts/train_nnue.py \
    --db data/games/jsonl_aggregated.db \
    --board-type square8 --num-players 2 \
    --epochs 50 --batch-size 2048 \
    --save-path models/nnue/square8_2p_new.pt'

# NNUE policy training with KL loss (when MCTS data available)
ssh gpu-primary 'cd ~/ringrift/ai-service && \
  venv/bin/python scripts/train_nnue_policy.py \
    --jsonl data/selfplay/mcts_square8_2p/games.jsonl \
    --auto-kl-loss --epochs 50'
```

### Run Model Tournament

```bash
ssh gpu-primary 'cd ~/ringrift/ai-service && \
  venv/bin/python scripts/run_model_elo_tournament.py \
    --board square8 --players 2 --games 50 \
    --include-nnue --run'
```

### Sync Models Across Cluster

```bash
ssh gpu-primary 'cd ~/ringrift/ai-service && \
  venv/bin/python scripts/sync_models.py --sync --use-sync-coordinator'
```

## Troubleshooting

### Low GPU Utilization

1. Check if workers are running:

   ```bash
   ssh $HOST 'ps aux | grep selfplay | grep -v grep'
   ```

2. Check for errors in logs:

   ```bash
   ssh $HOST 'tail -50 /tmp/gpu_selfplay.log'
   ```

3. Restart workers if needed:
   ```bash
   ssh $HOST 'pkill -f selfplay; sleep 5'
   # Then start new workers
   ```

### Node Unreachable

1. Check Tailscale:

   ```bash
   tailscale status
   ```

2. Try direct SSH:

   ```bash
   ssh -v $HOST
   ```

3. If still unreachable, contact cloud provider.

### Disk Full

1. Check usage:

   ```bash
   ssh $HOST 'df -h'
   ```

2. Find large files:

   ```bash
   ssh $HOST 'du -sh ~/ringrift/ai-service/data/*'
   ```

3. Clean up old data:
   ```bash
   ssh $HOST 'find ~/ringrift/ai-service/data/selfplay -name "*.jsonl" -mtime +7 -delete'
   ```

### P2P Orchestrator Crashes (FIXED - Dec 2025)

**Recent fixes deployed (commits 1270b64, dade90f, 6649601):**

**Problem: Missing dependencies**

```bash
# Check if dependencies are installed
ssh $HOST 'python3 -c "import aiohttp, psutil, yaml"'

# Install if missing
ssh $HOST 'cd ~/ringrift/ai-service && source venv/bin/activate && pip install aiohttp psutil pyyaml'
```

**Problem: Port 8770 already in use**

```bash
# Find what's using the port
ssh $HOST 'lsof -i :8770'

# Kill old P2P process
ssh $HOST 'pkill -f p2p_orchestrator'

# Wait 5 seconds, then restart
ssh $HOST 'sleep 5 && cd ~/ringrift/ai-service && PYTHONPATH=. nohup venv/bin/python scripts/p2p_orchestrator.py > logs/p2p.log 2>&1 &'
```

**Problem: Slow startup (loading state files)**

- P2P now has 120-second grace period (configurable via `RINGRIFT_P2P_STARTUP_GRACE_PERIOD`)
- No action needed - node_resilience will wait before checking

**Problem: Gossip endpoint 500 errors**

- Fixed: Gzip magic byte detection (0x1f 0x8b)
- Handles clients that set Content-Encoding: gzip but send raw JSON
- Update nodes to latest code: `python scripts/update_all_nodes.py`

**Problem: /dev/shm not accessible (macOS, permissions)**

- Fixed: Automatic fallback to disk storage
- No action needed - P2P will use temp directory

**Check P2P health:**

```bash
# Via HTTP
curl -s http://$HOST:8770/status | jq .

# Check logs
ssh $HOST 'tail -100 ~/ringrift/ai-service/logs/p2p.log'

# Check process
ssh $HOST 'ps aux | grep p2p_orchestrator | grep -v grep'
```

### Training Stuck

1. Check training log:

   ```bash
   ssh gpu-primary 'tail -100 /tmp/train_*.log'
   ```

2. Check GPU memory:

   ```bash
   ssh gpu-primary 'nvidia-smi'
   ```

3. Kill and restart if needed:
   ```bash
   ssh gpu-primary 'pkill -f train_nnue'
   ```

## Data Management

### Aggregate JSONL to Database

```bash
ssh gpu-primary 'cd ~/ringrift/ai-service && \
  venv/bin/python scripts/aggregate_jsonl_to_db.py \
    --input-dir data/selfplay \
    --output-db data/games/all_jsonl_training.db'
```

### Check Holdout Validation Set

```bash
# View holdout stats
cd ai-service
PYTHONPATH=. python scripts/holdout_validation.py --stats

# Evaluate model on holdout
PYTHONPATH=. python scripts/holdout_validation.py \
  --evaluate --model models/nnue/square8_2p.pt

# Check for overfitting
PYTHONPATH=. python scripts/holdout_validation.py --check-overfitting
```

### Backup to External Drive

```bash
# Backup models
rsync -avz ai-service/models/nnue/ /Volumes/RingRift-Data/model_backups/

# Backup training database
rsync -avz ai-service/data/games/merged_training.db /Volumes/RingRift-Data/db_backups/
```

## Alerting

### Test Cluster Alerting

```bash
cd ai-service
python scripts/health_alerting.py --check
```

### Run as Daemon

```bash
SLACK_WEBHOOK_URL="https://your-webhook-url" \
python scripts/health_alerting.py --daemon
```

## Logs

| Log         | Location                  | Description            |
| ----------- | ------------------------- | ---------------------- |
| Training    | `/tmp/train_*.log`        | NNUE training progress |
| Tournament  | `/tmp/elo_tournament.log` | Elo evaluation results |
| Selfplay    | `/tmp/gpu_selfplay.log`   | Worker output          |
| Aggregation | `/tmp/aggregation.log`    | Data aggregation       |

## Emergency Procedures

### Stop All Cluster Activity

```bash
# Replace with your actual node hostnames
for host in gpu-node-{1..10}; do
    echo "Stopping $host..."
    ssh $host 'pkill -f ringrift' 2>/dev/null || true
done
```

### Recovery After Outage

1. Check which nodes are back online:

   ```bash
   ./scripts/cluster_status.sh
   ```

2. Restart services on primary:

   ```bash
   ssh gpu-primary 'sudo systemctl restart ringrift-*'
   ```

3. Start P2P orchestrator and workers:

   ```bash
   # Start P2P orchestrator on primary node
   ssh gpu-primary 'cd ~/ringrift/ai-service && \
     PYTHONPATH=. nohup venv/bin/python scripts/p2p_orchestrator.py \
       --node-id gpu-node-1 --port 8770 \
       --ringrift-path ~/ringrift/ai-service > /tmp/p2p_orchestrator.log 2>&1 &'

   # Or use vast_lifecycle for automated worker management
   ssh gpu-primary 'cd ~/ringrift/ai-service && \
     venv/bin/python scripts/vast_lifecycle.py --start-jobs'
   ```

## Key Thresholds

| Metric          | Warning | Critical |
| --------------- | ------- | -------- |
| GPU Utilization | < 50%   | < 20%    |
| Disk Usage      | > 70%   | > 85%    |
| Worker Count    | < 5     | < 2      |
| Training Loss   | > 1.0   | > 2.0    |
| Holdout Gap     | > 0.10  | > 0.15   |

## Contacts

For infrastructure issues, check cloud provider status page first.
