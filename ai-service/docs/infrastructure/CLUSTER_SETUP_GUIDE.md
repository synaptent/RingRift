# RingRift Cluster Setup Guide

This guide walks you through setting up your own GPU cluster for distributed AI training.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Your Training Cluster                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │ gpu-primary  │    │ gpu-worker-1 │    │ gpu-worker-N │           │
│  │ (Training)   │    │ (Selfplay)   │    │ (Selfplay)   │           │
│  │ H100/A100    │    │ Any GPU      │    │ Any GPU      │           │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘           │
│         │                   │                   │                    │
│         └───────────────────┼───────────────────┘                    │
│                             │                                        │
│                     ┌───────▼───────┐                                │
│                     │  Data Sync    │                                │
│                     │  (rsync/SSH)  │                                │
│                     └───────┬───────┘                                │
│                             │                                        │
│                     ┌───────▼───────┐                                │
│                     │  Your Local   │                                │
│                     │   Machine     │                                │
│                     └───────────────┘                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Hardware Requirements

| Component | Minimum              | Recommended        |
| --------- | -------------------- | ------------------ |
| GPU       | RTX 3080 (10GB VRAM) | H100/GH200 (80GB+) |
| CPU       | 8 cores              | 32+ cores          |
| RAM       | 32GB                 | 128GB+             |
| Storage   | 100GB SSD            | 1TB+ NVMe          |
| Network   | 1Gbps                | 10Gbps+            |

### Software Requirements

- Ubuntu 22.04 LTS (recommended) or similar Linux
- Python 3.11+
- CUDA 12.0+ with cuDNN
- PyTorch 2.0+
- SSH server with key-based authentication

## Step 1: Prepare Your Nodes

### On each GPU node:

```bash
# Clone the repository
git clone https://github.com/an0mium/RingRift.git
cd RingRift/ai-service

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU is detected
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Set up SSH keys:

```bash
# On your local machine, generate a key if you don't have one
ssh-keygen -t ed25519 -f ~/.ssh/id_cluster

# Copy to each node
ssh-copy-id -i ~/.ssh/id_cluster ubuntu@<node-ip>
```

## Step 2: Configure Your Cluster

### Create config/sync_hosts.env:

```bash
cd ai-service
cp config/sync_hosts.env.example config/sync_hosts.env
```

Edit `config/sync_hosts.env`:

```bash
# Primary training node
SYNC_PRIMARY_HOST=ubuntu@10.0.0.1

# Fallback (optional)
SYNC_FALLBACK_HOST=ubuntu@10.0.0.2

# For sync_all_nodes.sh
COORDINATOR_IP="10.0.0.1"
COORDINATOR_PATH="/home/ubuntu/RingRift/ai-service/data/games"

# Worker nodes (space-separated)
NODES_STRING="worker1:10.0.0.10:/home/ubuntu/RingRift/ai-service/data/games worker2:10.0.0.11:/home/ubuntu/RingRift/ai-service/data/games"
```

### Create config/distributed_hosts.yaml:

```bash
cp config/distributed_hosts.template.yaml config/distributed_hosts.yaml
```

Edit with your actual hosts. Example:

```yaml
hosts:
  gpu-primary:
    ssh_host: '10.0.0.1'
    ssh_user: 'ubuntu'
    ringrift_path: '~/RingRift/ai-service'
    role: 'training,tournament'
    gpu: 'H100'

  gpu-worker-1:
    ssh_host: '10.0.0.10'
    ssh_user: 'ubuntu'
    ringrift_path: '~/RingRift/ai-service'
    role: 'selfplay'
    gpu: 'RTX 4090'
```

If you use `unified_data_sync.py`, also create `config/remote_hosts.yaml`
(legacy sync format) with the same hosts.

## Step 3: Deploy Code to Cluster

```bash
# From your local machine
python scripts/update_cluster_code.py --auto-stash
```

Or manually:

```bash
for host in gpu-primary gpu-worker-1 gpu-worker-2; do
  rsync -avz --exclude 'venv' --exclude '__pycache__' --exclude 'data' \
    ./ $host:~/RingRift/ai-service/
done
```

## Step 4: Start Training

### Option A: Unified AI Loop (Recommended)

The unified loop handles everything automatically:

```bash
# On your primary node
ssh gpu-primary 'cd ~/RingRift/ai-service && \
  source venv/bin/activate && \
  nohup python scripts/unified_ai_loop.py --start > logs/loop.log 2>&1 &'

# Check status
ssh gpu-primary 'cd ~/RingRift/ai-service && python scripts/unified_ai_loop.py --status'
```

### Option B: Manual Control

Start components individually:

```bash
# Start selfplay on worker nodes
ssh gpu-worker-1 'cd ~/RingRift/ai-service && source venv/bin/activate && \
  nohup python scripts/run_gpu_selfplay.py \
    --board-type square8 --num-players 2 --num-games 10000 \
    --output-dir data/selfplay/worker1 > logs/selfplay.log 2>&1 &'

# Start training on primary node
ssh gpu-primary 'cd ~/RingRift/ai-service && source venv/bin/activate && \
  python scripts/train_nnue_policy.py \
    --db data/games/selfplay.db --board square8 --num-players 2 \
    --epochs 50 --auto-kl-loss'
```

## Step 5: Set Up Automated Data Sync

Install cron jobs on your local machine:

```bash
crontab config/crontab_training.txt
```

This sets up:

- Data sync every 15 minutes
- Model pruning daily at 3am
- Health checks every 30 minutes

## Step 6: Monitor Progress

### Check training status:

```bash
# Game counts by config
sqlite3 data/games/selfplay.db \
  "SELECT board_type, num_players, COUNT(*) FROM games GROUP BY 1,2"

# Model counts
ls models/nnue/*.pt | wc -l

# Active processes on cluster
for host in gpu-primary gpu-worker-1; do
  echo "=== $host ==="
  ssh $host 'ps aux | grep -E "train|selfplay" | grep -v grep'
done
```

### View logs:

```bash
# Training logs
tail -f logs/unified_loop/daemon.log

# Selfplay logs
ssh gpu-worker-1 'tail -f ~/RingRift/ai-service/logs/selfplay.log'
```

## Troubleshooting

### SSH Connection Issues

```bash
# Test connectivity
ssh -o ConnectTimeout=5 gpu-primary 'echo OK'

# Check SSH key
ssh-add -l

# Verify host fingerprint
ssh-keygen -R <ip> && ssh <ip>
```

### GPU Not Detected

```bash
# Check CUDA
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory

```bash
# Reduce batch size
python scripts/train_nnue_policy.py --batch-size 512

# Use gradient checkpointing
python scripts/train_nnue_policy.py --gradient-checkpointing
```

### Training Not Triggering

```bash
# Check loop state
cat logs/unified_loop/unified_loop_state.json | python -m json.tool

# Reset and restart
python scripts/unified_ai_loop.py --stop
rm logs/unified_loop/unified_loop_state.json
python scripts/unified_ai_loop.py --start
```

## Network Options

### Option 1: Direct IPs (Simple)

Use direct IP addresses in your configs. Works well for:

- Same datacenter/LAN
- Cloud VPCs

### Option 2: Tailscale (Recommended)

[Tailscale](https://tailscale.com/) provides:

- Encrypted mesh networking
- NAT traversal
- Stable IPs across networks

```bash
# Install on each node
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Use 100.x.x.x IPs in your config
```

### Option 3: SSH Tunnels

For nodes behind strict firewalls:

```bash
# Create reverse tunnel
ssh -R 8770:localhost:8770 relay-server

# See scripts/autossh_p2p_tunnel.sh for automation
```

## Scaling Up

### Adding More Workers

1. Set up the new node (Step 1)
2. Add to `config/distributed_hosts.yaml` (and `config/remote_hosts.yaml` if using data sync)
3. Deploy code: `python scripts/update_cluster_code.py`
4. Start selfplay on the new node

### Multi-Config Training

Configure multi-board training in `config/unified_loop.yaml` and run the unified loop:

```bash
python scripts/unified_ai_loop.py --start --config config/unified_loop.yaml
```

### Curriculum Training

Progressive training from simple to complex:

```bash
python scripts/curriculum_training.py \
  --auto-progress \
  --board square8 --num-players 2 \
  --db data/games/selfplay.db
```

## See Also

- [TRAINING_PIPELINE.md](../training/TRAINING_PIPELINE.md) - Detailed pipeline documentation
- [UNIFIED_AI_LOOP.md](../training/UNIFIED_AI_LOOP.md) - Unified loop reference
- [DISTRIBUTED_SELFPLAY.md](../training/DISTRIBUTED_SELFPLAY.md) - Remote worker setup
