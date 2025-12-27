# Cluster Deployment Runbook

This runbook covers the complete process for deploying and managing the RingRift distributed training cluster, from initial setup to health verification.

**Last Updated:** December 2025
**Target Audience:** DevOps engineers, cluster administrators

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Node Configuration](#node-configuration)
3. [P2P Network Initialization](#p2p-network-initialization)
4. [Model Distribution](#model-distribution)
5. [Health Verification](#health-verification)
6. [Common Issues & Troubleshooting](#common-issues--troubleshooting)
7. [Maintenance Operations](#maintenance-operations)

---

## Prerequisites

### Required Software

#### On Control Node (Coordinator)

```bash
# Python 3.10+ with dependencies
python3 --version  # Should be 3.10 or higher

# Install core dependencies
cd ai-service
pip install -r requirements.txt

# Verify PyYAML for config parsing
python3 -c "import yaml; print('PyYAML OK')"

# Verify aiohttp for P2P communication
python3 -c "import aiohttp; print('aiohttp OK')"

# SSH client (usually pre-installed on Linux/macOS)
ssh -V
```

#### On Worker Nodes

```bash
# Python 3.10+ with PyTorch
python3 --version

# Virtual environment (recommended)
python3 -m venv ~/ringrift/ai-service/venv
source ~/ringrift/ai-service/venv/bin/activate

# Install dependencies
cd ~/ringrift/ai-service
pip install -r requirements.txt

# Verify PyTorch installation
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check CUDA availability (GPU nodes only)
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Verify nvidia-smi for GPU monitoring (GPU nodes only)
nvidia-smi
```

### SSH Key Setup

All nodes must be accessible via passwordless SSH from the coordinator.

#### Generate SSH Keys (if needed)

```bash
# On coordinator node
ssh-keygen -t ed25519 -f ~/.ssh/id_cluster -C "ringrift-cluster"

# Or use existing keys
# Common key paths: ~/.ssh/id_ed25519, ~/.ssh/id_rsa, ~/.ssh/id_cluster
```

#### Distribute Public Keys

```bash
# Copy to each worker node
ssh-copy-id -i ~/.ssh/id_cluster.pub user@worker-node-1
ssh-copy-id -i ~/.ssh/id_cluster.pub user@worker-node-2

# Test connectivity
ssh -i ~/.ssh/id_cluster user@worker-node-1 "hostname"
```

#### SSH Config (Optional)

Create `~/.ssh/config` for simplified access:

```
Host worker-*
    User ubuntu
    IdentityFile ~/.ssh/id_cluster
    StrictHostKeyChecking no
    ConnectTimeout 10
```

### Network Requirements

#### Required Ports

| Port | Protocol | Purpose                          | Required On |
| ---- | -------- | -------------------------------- | ----------- |
| 22   | TCP      | SSH                              | All nodes   |
| 8770 | TCP      | P2P orchestrator HTTP API        | All nodes   |
| 8765 | TCP      | P2P peer-to-peer communication   | All nodes   |
| 8767 | TCP      | Model upload/distribution (HTTP) | All nodes   |
| 8080 | TCP      | Health check endpoint (optional) | All nodes   |

#### Firewall Configuration

```bash
# On each worker node (example for Ubuntu with ufw)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8770/tcp  # P2P HTTP
sudo ufw allow 8765/tcp  # P2P gossip
sudo ufw allow 8767/tcp  # Model distribution
sudo ufw enable

# Or using iptables
sudo iptables -A INPUT -p tcp --dport 8770 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8765 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8767 -j ACCEPT
```

#### Network Connectivity Test

```bash
# From coordinator, test each worker node
for host in worker-1 worker-2 worker-3; do
    echo "Testing $host..."
    nc -zv $host 22    # SSH
    nc -zv $host 8770  # P2P HTTP (if already running)
done
```

### GPU Driver Requirements (GPU Nodes Only)

```bash
# Check NVIDIA driver version
nvidia-smi

# Minimum recommended: NVIDIA driver 525+ for CUDA 12
# For older CUDA 11.8, driver 450+ is sufficient

# Check CUDA version
nvcc --version

# Verify PyTorch can see GPUs
python3 -c "import torch; print(f'CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"
```

If GPU not detected:

- Verify driver installation: `sudo apt install nvidia-driver-535`
- Check CUDA toolkit: `sudo apt install cuda-toolkit-12-1`
- Restart node: `sudo reboot`

---

## Node Configuration

### Create `distributed_hosts.yaml`

Copy the template and configure your cluster nodes:

```bash
cd ai-service
cp config/distributed_hosts.template.yaml config/distributed_hosts.yaml
```

### Configuration Schema

Edit `config/distributed_hosts.yaml`:

```yaml
hosts:
  # Example: Nebius H100 GPU node
  lambda-gh200-a:
    ssh_host: '100.88.35.19' # Tailscale IP or public IP
    ssh_user: 'ubuntu' # SSH username
    ssh_key: '~/.ssh/id_cluster' # Path to SSH private key
    ssh_port: 22 # SSH port (default: 22)
    ringrift_path: '~/ringrift/ai-service' # Path to ai-service on remote
    venv_activate: 'source ~/ringrift/ai-service/venv/bin/activate'
    status: 'ready' # ready, disabled, maintenance
    role: 'mixed' # selfplay, mixed, nn_training, coordinator
    cpus: 32 # CPU cores (for scheduling)
    memory_gb: 128 # RAM in GB
    gpu: 'GH200 96GB' # GPU name (empty for CPU nodes)
    max_parallel_jobs: 4 # Concurrency limit

    # Optional: Tailscale IP for private network
    tailscale_ip: '100.88.35.19'

  # Example: Vast.ai ephemeral GPU node
  vast-rtx5090-1:
    ssh_host: '38.128.233.145'
    ssh_user: 'root'
    ssh_key: '~/.ssh/id_ed25519'
    ssh_port: 33085 # Vast.ai uses custom ports
    ringrift_path: '/workspace/ringrift/ai-service'
    venv_activate: 'source /workspace/ringrift/ai-service/venv/bin/activate'
    status: 'ready'
    role: 'selfplay' # Ephemeral nodes best for selfplay
    memory_gb: 64
    gpu: 'RTX 5090 24GB'
    max_parallel_jobs: 2

  # Example: CPU-only node (Hetzner)
  hetzner-cpu-1:
    ssh_host: '168.119.134.85'
    ssh_user: 'root'
    ssh_key: '~/.ssh/id_ed25519'
    ringrift_path: '/root/ringrift/ai-service'
    venv_activate: 'source /root/ringrift/ai-service/venv/bin/activate'
    status: 'ready'
    role: 'selfplay' # CPU selfplay only
    cpus: 16
    memory_gb: 32
    gpu: '' # Empty for CPU-only
    max_parallel_jobs: 2
```

### Node Roles

| Role          | Description                    | Recommended Hardware                   |
| ------------- | ------------------------------ | -------------------------------------- |
| `coordinator` | Control plane only, no workers | Any (runs on your laptop/control node) |
| `selfplay`    | Self-play game generation      | GPU: 8GB+ VRAM, CPU: 8GB+ RAM          |
| `nn_training` | Neural network training        | GPU: 16GB+ VRAM, 32GB+ RAM             |
| `mixed`       | Both selfplay and training     | GPU: 24GB+ VRAM, 64GB+ RAM             |

### Environment Variables (Optional)

Set these on worker nodes to tune behavior:

```bash
# Add to ~/.bashrc or systemd service file

# Process management
export RINGRIFT_JOB_GRACE_PERIOD=60              # Seconds before SIGKILL after SIGTERM
export RINGRIFT_GPU_IDLE_THRESHOLD=600           # Seconds of GPU idle before killing stuck processes
export RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD=128  # Max selfplay processes per node

# P2P network
export RINGRIFT_ADVERTISE_HOST=100.88.35.19      # Override auto-detected IP for P2P
export RINGRIFT_P2P_PORT=8770                    # P2P HTTP port
export RINGRIFT_P2P_GOSSIP_PORT=8765             # P2P gossip port

# Cluster authentication (optional)
export RINGRIFT_CLUSTER_AUTH_TOKEN=your-secret-token
```

### Verify Configuration

```bash
# On coordinator, test SSH connectivity to all nodes
python3 scripts/cluster_health_check.py

# Or test manually
python3 -c "
from app.distributed.hosts import load_hosts_config
config = load_hosts_config()
print(f'Loaded {len(config.get(\"hosts\", {}))} hosts')
for name, info in config.get('hosts', {}).items():
    print(f'  - {name}: {info.get(\"ssh_host\")} ({info.get(\"role\")})')
"
```

---

## P2P Network Initialization

The P2P mesh network provides decentralized coordination without a central control server.

### Leader Election & Voter Quorum

The P2P cluster uses **Raft consensus** with designated voter nodes for leader election.

#### Configure Voters (Important!)

Set voter nodes in `ai-service/config/distributed_hosts.yaml` (authoritative):

```yaml
p2p_voters:
  - nebius-backbone-1
  - nebius-h100-3
  - hetzner-cpu1
  - hetzner-cpu2
  - vultr-a100-20gb
```

**Voter Selection Criteria:**

- Choose 3-5 stable nodes (not ephemeral Vast.ai instances)
- Prefer nodes with high uptime (Nebius, Hetzner, Vultr persistent instances)
- Avoid nodes that may be terminated frequently
- Quorum requires `(n/2) + 1` voters online (e.g., 3 of 5)

### Start P2P on All Nodes

Use the automated script to start P2P daemons cluster-wide:

```bash
# Start P2P on all nodes
cd ai-service
python3 scripts/start_p2p_cluster.py

# Check status only (don't start)
python3 scripts/start_p2p_cluster.py --check

# Restart all nodes (fixes IP changes)
python3 scripts/start_p2p_cluster.py --restart

# Start on specific node
python3 scripts/start_p2p_cluster.py --node nebius-backbone-1
```

**Expected Output:**

```
Processing 43 hosts (starting P2P)
------------------------------------------------------------
Results:
  ✓ nebius-backbone-1: Started (PID 12345, advertise=89.169.110.128)
  ✓ nebius-h100-3: Started (PID 12346, advertise=89.169.111.139)
  ✓ vast-rtx5090-1: Started (PID 12347, advertise=38.128.233.145)
  ...
  ✗ hetzner-cpu-1: Connection timeout
  ✓ runpod-h100: Already running

Summary: 38 running, 5 newly started, 0 restarted, 2 failed
```

### Manual P2P Start (Alternative)

If the automated script fails, start P2P manually on each node:

```bash
# SSH to worker node
ssh -i ~/.ssh/id_cluster ubuntu@worker-node-1

# Navigate to ai-service
cd ~/ringrift/ai-service
source venv/bin/activate

# Start P2P with nohup
export RINGRIFT_ADVERTISE_HOST=$(hostname -I | awk '{print $1}')
PYTHONPATH=. nohup venv/bin/python scripts/p2p_orchestrator.py --node-id $(hostname) --port 8770 --peers <coordinator_urls> > logs/p2p.log 2>&1 &

# Verify it's running
pgrep -f p2p_orchestrator
curl -s http://localhost:8770/status | python3 -c 'import sys,json; print(json.load(sys.stdin).get("leader_id"))'
```

### Verify P2P Cluster Formation

```bash
# Check cluster status from any node
curl -s http://localhost:8770/status | python3 -c '
import sys, json
d = json.load(sys.stdin)
print(f"Leader: {d.get(\"leader_id\")}")
print(f"Alive peers: {d.get(\"alive_peers\")}")
print(f"Total nodes: {len(d.get(\"nodes\", []))}")
'

# Or use the cluster monitor
python3 -m app.distributed.cluster_monitor
```

**Healthy Cluster Output:**

```
Leader: nebius-backbone-1
Alive peers: 38
Total nodes: 43
```

### Leader Election Troubleshooting

If no leader is elected:

```bash
# Check voter quorum
python3 -c "
voters = ['nebius-backbone-1', 'nebius-h100-3', 'hetzner-cpu1', 'hetzner-cpu2', 'vultr-a100-20gb']
required = (len(voters) // 2) + 1
print(f'Voter quorum: {required} of {len(voters)} voters required')
"

# Verify voters are online
for node in nebius-backbone-1 nebius-h100-3 hetzner-cpu1; do
    echo "Checking $node..."
    ssh $node "pgrep -f p2p_orchestrator && curl -s http://localhost:8770/status" || echo "DOWN"
done

# Force restart voters (if needed)
python3 scripts/start_p2p_cluster.py --restart --node nebius-backbone-1
```

---

## Model Distribution

After training completes, models must be distributed to all nodes for selfplay and evaluation.

### Automatic Model Distribution

The `ModelDistributionDaemon` automatically syncs models after promotion:

```bash
# Verify daemon is running
python3 scripts/launch_daemons.py --status | grep MODEL_DISTRIBUTION

# If not running, start it
python3 scripts/launch_daemons.py --daemon MODEL_DISTRIBUTION
```

**How it works:**

1. Subscribes to `MODEL_PROMOTED` events from the event router
2. Uses HTTP distribution (preferred) or rsync (fallback) to sync to all nodes
3. Verifies SHA256 checksums before and after transfer
4. Emits `MODEL_DISTRIBUTION_COMPLETE` event when done
5. Updates `ClusterManifest` to track model locations

### Manual Model Distribution

If automatic distribution fails or you need to sync models immediately:

```bash
# Distribute a specific model to all nodes
cd ai-service
python3 scripts/sync_models.py --distribute --model models/canonical_hex8_2p.pth

# Distribute all canonical models
python3 scripts/sync_models.py --distribute --all-canonical

# Verify distribution (checks checksums on remote nodes)
python3 scripts/sync_models.py --verify --model models/canonical_hex8_2p.pth
```

### Create `ringrift_best_*` Symlinks

The system expects models to be named with the `ringrift_best_*` pattern:

```bash
# On each node, create symlinks
cd ~/ringrift/ai-service/models

# Example: Create symlinks for all canonical models
for board in hex8 square8 square19 hexagonal; do
    for players in 2 3 4; do
        canonical="canonical_${board}_${players}p.pth"
        symlink="ringrift_best_${board}_${players}p.pth"

        if [ -f "$canonical" ]; then
            ln -sf "$canonical" "$symlink"
            echo "Created $symlink -> $canonical"
        fi
    done
done

# Verify symlinks
ls -lh ringrift_best_*
```

### Verify Model Integrity

Check that models are correctly distributed and have valid checksums:

```bash
# Check model checksums on all nodes
python3 scripts/validate_models.py --all-nodes

# Or manually verify on a single node
ssh worker-node-1 "
cd ~/ringrift/ai-service/models
sha256sum canonical_hex8_2p.pth
"
```

### Model Storage Requirements

Ensure nodes have sufficient disk space:

| Board Type | 2-Player | 3-Player | 4-Player |
| ---------- | -------- | -------- | -------- |
| hex8       | 40 MB    | 40 MB    | 40 MB    |
| square8    | 96 MB    | 15 MB    | 96 MB    |
| square19   | 107 MB   | 108 MB   | 108 MB   |
| hexagonal  | 174 MB   | 174 MB   | 174 MB   |

**Total for all 12 canonical models:** ~1.1 GB

---

## Health Verification

### Cluster-Wide Health Check

```bash
# Comprehensive health check across all nodes
python3 scripts/cluster_health_check.py

# Check specific nodes
python3 scripts/cluster_health_check.py --nodes lambda-gh200-a lambda-gh200-b

# Attempt to fix common issues (install dependencies, etc.)
python3 scripts/cluster_health_check.py --fix
```

**Expected Output:**

```
✅ lambda-gh200-a: reachable, pytorch_ok, cuda_available (1 GPU), disk: 450GB free
✅ lambda-gh200-b: reachable, pytorch_ok, cuda_available (1 GPU), disk: 480GB free
⚠️ vast-rtx5090-1: reachable, pytorch_ok, cuda_available (1 GPU), disk: 50GB free (low!)
❌ hetzner-cpu-1: unreachable (SSH timeout)
```

### P2P Connectivity Check

```bash
# Check P2P status on all nodes
python3 scripts/start_p2p_cluster.py --check

# Monitor cluster in real-time
python3 -m app.distributed.cluster_monitor --watch --interval 10
```

### Component Health Checks

```bash
# Run health checks for all pipeline components
python3 -c "
from app.distributed.health_checks import get_health_summary, format_health_report

summary = get_health_summary()
print(format_health_report(summary))
"
```

**Components Checked:**

- `data_sync` - Data synchronization daemon
- `training` - Training pipeline status
- `evaluation` - Tournament/gauntlet evaluation
- `coordinator` - Cluster coordinator tasks
- `coordinator_managers` - Coordinator manager metrics
- `resources` - CPU, memory, disk usage

### GPU Health Verification (GPU Nodes)

```bash
# Verify GPU availability on all GPU nodes
for node in $(python3 -c "
from app.distributed.hosts import load_hosts_config
config = load_hosts_config()
for name, info in config.get('hosts', {}).items():
    if info.get('gpu'):
        print(name)
"); do
    echo "Checking GPU on $node..."
    ssh $node "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader"
done
```

### Database Integrity Check

```bash
# Verify game databases on all nodes
python3 scripts/db_health_check.py --all-nodes

# Check specific database
python3 scripts/db_health_check.py --db data/games/selfplay.db
```

### Monitoring Dashboard

For continuous monitoring, use the cluster monitor in watch mode:

```bash
# Real-time cluster dashboard (updates every 10 seconds)
python3 -m app.distributed.cluster_monitor --watch --interval 10
```

---

## Common Issues & Troubleshooting

### Node Not Joining Cluster

**Symptoms:**

- Node shows as `unreachable` in `cluster_monitor`
- P2P status shows 0 peers

**Diagnosis:**

```bash
# Check if P2P process is running
ssh worker-node "pgrep -f p2p_orchestrator || echo 'Not running'"

# Check P2P logs for errors
ssh worker-node "tail -50 ~/ringrift/ai-service/logs/p2p.log"

# Check firewall/ports
ssh worker-node "netstat -tuln | grep -E '8770|8765'"
```

**Solutions:**

1. **Firewall blocking ports:**

   ```bash
   ssh worker-node "sudo ufw allow 8770/tcp && sudo ufw allow 8765/tcp"
   ```

2. **Wrong advertise IP:**

   ```bash
   # Restart with correct IP
   python3 scripts/start_p2p_cluster.py --restart --node worker-node-name
   ```

3. **Network connectivity issues:**

   ```bash
   # Test connectivity from coordinator to worker
   nc -zv worker-node 8770

   # Test reverse connectivity
   ssh worker-node "nc -zv coordinator-ip 8770"
   ```

4. **Clock skew (affects Raft consensus):**
   ```bash
   # Sync time on worker node
   ssh worker-node "sudo ntpdate pool.ntp.org"
   # Or use systemd-timesyncd
   ssh worker-node "sudo timedatectl set-ntp true"
   ```

### SSH Connection Failures

**Symptoms:**

- `Permission denied (publickey)`
- `Connection timeout`
- `Host key verification failed`

**Solutions:**

1. **Missing SSH key:**

   ```bash
   # Copy public key to worker
   ssh-copy-id -i ~/.ssh/id_cluster.pub user@worker-node
   ```

2. **Wrong SSH key path in config:**

   ```yaml
   # Fix distributed_hosts.yaml
   ssh_key: '~/.ssh/id_cluster' # Ensure this matches your actual key
   ```

3. **SSH key permissions:**

   ```bash
   chmod 600 ~/.ssh/id_cluster
   chmod 644 ~/.ssh/id_cluster.pub
   ```

4. **Firewall blocking SSH:**

   ```bash
   ssh worker-node "sudo ufw allow 22/tcp"
   ```

5. **Custom SSH port:**
   ```yaml
   # Update distributed_hosts.yaml
   ssh_port: 33085 # For Vast.ai or custom ports
   ```

### Model Sync Failures

**Symptoms:**

- `ModelDistributionDaemon` shows errors
- Selfplay fails with "model not found"
- Checksum verification failures

**Diagnosis:**

```bash
# Check model distribution logs
python3 scripts/launch_daemons.py --logs MODEL_DISTRIBUTION

# Manually test model sync to a node
python3 scripts/sync_models.py --distribute --model models/canonical_hex8_2p.pth --host worker-node

# Check remote model directory
ssh worker-node "ls -lh ~/ringrift/ai-service/models/"
```

**Solutions:**

1. **Insufficient disk space on worker:**

   ```bash
   # Check disk space
   ssh worker-node "df -h ~/ringrift/ai-service"

   # Clean up old models/data
   ssh worker-node "cd ~/ringrift/ai-service && python3 scripts/cleanup_old_data.py"
   ```

2. **rsync not installed:**

   ```bash
   ssh worker-node "sudo apt-get install -y rsync"
   ```

3. **Model file corrupted:**

   ```bash
   # Verify checksum on source
   sha256sum models/canonical_hex8_2p.pth

   # Re-download or re-train
   python3 scripts/sync_models.py --force --model models/canonical_hex8_2p.pth
   ```

4. **Missing symlinks:**
   ```bash
   # Recreate symlinks on all nodes
   python3 scripts/sync_models.py --create-symlinks
   ```

### High Resource Usage

**Symptoms:**

- `resources` component shows warnings in health check
- Nodes become unresponsive
- OOM (out of memory) errors

**Diagnosis:**

```bash
# Check resource usage across cluster
python3 -m app.distributed.cluster_monitor

# Check specific node resources
ssh worker-node "top -b -n 1 | head -20"
ssh worker-node "free -h"
ssh worker-node "df -h"
```

**Solutions:**

1. **Too many parallel jobs:**

   ```yaml
   # Reduce in distributed_hosts.yaml
   max_parallel_jobs: 2 # Lower from 4
   ```

2. **Memory leak in selfplay:**

   ```bash
   # Restart stuck processes
   ssh worker-node "pkill -f selfplay.py"

   # Set process limits
   export RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD=64
   ```

3. **Disk space critical:**

   ```bash
   # Clean up old training data
   ssh worker-node "cd ~/ringrift/ai-service && python3 scripts/cleanup_old_data.py --keep-days 7"

   # Archive old databases
   ssh worker-node "cd ~/ringrift/ai-service/data/games && tar czf old_$(date +%Y%m%d).tar.gz *.db && rm *.db"
   ```

### Leader Election Failures

**Symptoms:**

- P2P status shows `leader_id: null` or `leader_id: ?`
- Cluster operations fail with "no leader available"

**Diagnosis:**

```bash
# Check voter status
for voter in nebius-backbone-1 nebius-h100-3 hetzner-cpu1 hetzner-cpu2 vultr-a100-20gb; do
    echo "=== $voter ==="
    ssh $voter "curl -s http://localhost:8770/status | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f\"Role: {d.get(\\\"role\\\")}, Leader: {d.get(\\\"leader_id\\\")}\")' 2>&1" || echo "UNREACHABLE"
done
```

**Solutions:**

1. **Not enough voters online:**

   ```bash
   # Start P2P on missing voters
   python3 scripts/start_p2p_cluster.py --node nebius-h100-3
   python3 scripts/start_p2p_cluster.py --node nebius-backbone-1
   ```

2. **Network partition:**

   ```bash
   # Restart all P2P nodes to force re-election
   python3 scripts/start_p2p_cluster.py --restart
   ```

3. **Update voter list (if voters changed):**

   Update `p2p_voters` in `ai-service/config/distributed_hosts.yaml`.

   Then restart all P2P nodes:

   ```bash
   python3 scripts/start_p2p_cluster.py --restart
   ```

### Database Corruption

**Symptoms:**

- `data_sync` component fails health check
- `sqlite3.DatabaseError: database disk image is malformed`

**Solutions:**

```bash
# Verify database integrity
ssh worker-node "sqlite3 ~/ringrift/ai-service/data/games/selfplay.db 'PRAGMA integrity_check;'"

# Export to SQL and rebuild
ssh worker-node "
cd ~/ringrift/ai-service/data/games
sqlite3 selfplay.db .dump > backup.sql
mv selfplay.db selfplay.db.corrupt
sqlite3 selfplay.db < backup.sql
"

# Or restore from another node
rsync -avz worker-node-2:~/ringrift/ai-service/data/games/selfplay.db data/games/
```

---

## Maintenance Operations

### Adding New Nodes

1. **Configure node in `distributed_hosts.yaml`:**

   ```yaml
   new-node-name:
     ssh_host: '192.168.1.100'
     ssh_user: 'ubuntu'
     ssh_key: '~/.ssh/id_cluster'
     ringrift_path: '~/ringrift/ai-service'
     venv_activate: 'source ~/ringrift/ai-service/venv/bin/activate'
     status: 'ready'
     role: 'mixed'
     memory_gb: 64
     gpu: 'RTX 4090'
   ```

2. **Install dependencies on new node:**

   ```bash
   ssh new-node-name "
   cd ~/ringrift/ai-service
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   "
   ```

3. **Start P2P:**

   ```bash
   python3 scripts/start_p2p_cluster.py --node new-node-name
   ```

4. **Sync models:**

   ```bash
   python3 scripts/sync_models.py --distribute --host new-node-name
   ```

5. **Verify:**
   ```bash
   python3 -m app.distributed.cluster_monitor | grep new-node-name
   ```

### Removing Nodes

1. **Mark as disabled in `distributed_hosts.yaml`:**

   ```yaml
   old-node:
     status: 'disabled' # Change from 'ready'
   ```

2. **Stop P2P on node:**

   ```bash
   ssh old-node "pkill -f p2p_orchestrator"
   ```

3. **Remove from voter list (if applicable):**
   Remove the node ID from `p2p_voters` in `ai-service/config/distributed_hosts.yaml` and restart P2P on remaining voters.

### Updating Code on All Nodes

```bash
# Update code on all nodes
python3 scripts/update_cluster_code.py

# Or manually with git
for node in $(python3 -c "
from app.distributed.hosts import load_hosts_config
config = load_hosts_config()
for name in config.get('hosts', {}).keys():
    print(name)
"); do
    echo "Updating $node..."
    ssh $node "cd ~/ringrift/ai-service && git pull && source venv/bin/activate && pip install -r requirements.txt"
done
```

### Restarting P2P Cluster

```bash
# Graceful restart (one node at a time)
python3 scripts/start_p2p_cluster.py --restart

# Or restart all at once (faster but causes brief downtime)
for node in $(python3 -c "
from app.distributed.hosts import load_hosts_config
config = load_hosts_config()
for name in config.get('hosts', {}).keys():
    print(name)
"); do
    ssh $node "pkill -f p2p_orchestrator" &
done
wait

sleep 5
python3 scripts/start_p2p_cluster.py
```

### Backup and Restore

#### Backup Game Databases

```bash
# Backup from all nodes
python3 scripts/aggregate_cluster_dbs.py --output backup_$(date +%Y%m%d).tar.gz

# Or manual backup
for node in worker-1 worker-2 worker-3; do
    rsync -avz $node:~/ringrift/ai-service/data/games/*.db backup/$node/
done
```

#### Restore Models

```bash
# Restore models to all nodes
python3 scripts/sync_models.py --distribute --all-canonical

# Or from backup
for node in worker-1 worker-2 worker-3; do
    rsync -avz backup/models/*.pth $node:~/ringrift/ai-service/models/
done
```

---

## Additional Resources

- **Runbooks:**
  - [Troubleshooting Guide](TROUBLESHOOTING.md)
  - [Cluster Health Critical](CLUSTER_HEALTH_CRITICAL.md)
  - [Sync Host Critical](SYNC_HOST_CRITICAL.md)

- **Scripts:**
  - `scripts/cluster_health_check.py` - Comprehensive health checks
  - `scripts/start_p2p_cluster.py` - P2P orchestrator management
  - `scripts/sync_models.py` - Model distribution
  - `app/distributed/cluster_monitor.py` - Real-time monitoring

- **Configuration:**
  - `config/distributed_hosts.yaml` - Cluster node configuration
  - `config/distributed_hosts.template.yaml` - Configuration template

---

**Questions or Issues?**

See [docs/runbooks/TROUBLESHOOTING.md](TROUBLESHOOTING.md) or check the health status:

```bash
python3 -c "
from app.distributed.health_checks import get_health_summary, format_health_report
summary = get_health_summary()
print(format_health_report(summary))
"
```
