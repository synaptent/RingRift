# RingRift Cluster Node Configuration Guide

This document describes how to configure new nodes to join the RingRift P2P training cluster.

## Prerequisites

- Ubuntu 20.04+ or compatible Linux distribution
- Tailscale installed and connected to the RingRift network
- SSH access with key authentication
- Python 3.10+
- NVIDIA GPU with CUDA drivers (for GPU nodes)

## Node Types

| Type                  | Role                           | Example Nodes               |
| --------------------- | ------------------------------ | --------------------------- |
| **Lambda H100/GH200** | Primary NN training            | lambda-h100, lambda-gh200-a |
| **Lambda A10**        | Secondary NN training          | lambda-a10                  |
| **AWS Instances**     | Selfplay, CMA-ES, coordination | aws-staging, aws-proxy      |
| **Vast.ai**           | GPU selfplay, CPU-bound tasks  | vast-5080, vast-3070-b      |

## Configuration Steps

### 1. Install Tailscale

```bash
# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Authenticate (requires admin approval)
sudo tailscale up --hostname=<node-name>

# Verify connection
tailscale status
```

### 2. Clone Repository

```bash
# For Lambda/AWS nodes (ubuntu user)
cd ~
git clone https://github.com/an0mium/RingRift.git ringrift

# For Vast nodes (root user)
cd /root
git clone https://github.com/an0mium/RingRift.git ringrift
```

### 3. Set Up Python Virtual Environment

```bash
cd ~/ringrift/ai-service  # or /root/ringrift/ai-service for Vast

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create Node Configuration

Create `/etc/ringrift/node.conf`:

```bash
sudo mkdir -p /etc/ringrift
sudo tee /etc/ringrift/node.conf << 'EOF'
NODE_ID=<node-name>
P2P_PORT=8770
RINGRIFT_PATH=/home/ubuntu/ringrift  # or /root/ringrift for Vast
COORDINATOR_URL=https://p2p.ringrift.ai,http://100.78.101.123:8770,http://100.88.176.74:8770
RINGRIFT_P2P_VOTERS=aws-staging,lambda-a10,lambda-h100,lambda-2xh100,lambda-gh200-a,lambda-gh200-b,lambda-gh200-c,lambda-gh200-d,lambda-gh200-e,lambda-gh200-f
EOF
```

**Configuration Values:**

| Variable              | Description                                              |
| --------------------- | -------------------------------------------------------- |
| `NODE_ID`             | Unique identifier for this node (e.g., `lambda-gh200-g`) |
| `P2P_PORT`            | Port for P2P communication (default: 8770)               |
| `RINGRIFT_PATH`       | Path to the ringrift repository                          |
| `COORDINATOR_URL`     | Comma-separated list of coordinator endpoints            |
| `RINGRIFT_P2P_VOTERS` | Comma-separated list of voter node IDs                   |

### 5. Install Systemd Service (Lambda/AWS only)

```bash
# Create log directory
sudo mkdir -p /var/log/ringrift

# Copy service file
sudo cp ~/ringrift/ai-service/deploy/systemd/ringrift-p2p-universal.service \
    /etc/systemd/system/ringrift-p2p.service

# Reload and enable
sudo systemctl daemon-reload
sudo systemctl enable ringrift-p2p.service
sudo systemctl start ringrift-p2p.service

# Verify
systemctl status ringrift-p2p.service
```

### 6. Start P2P (Vast.ai - No Systemd)

Vast containers don't have systemd. Start P2P manually:

```bash
# Kill existing process
pkill -f p2p_orchestrator.py 2>/dev/null

# Start in background
cd /root/ringrift/ai-service
PYTHONPATH=/root/ringrift/ai-service nohup venv/bin/python scripts/p2p_orchestrator.py \
    --node-id <node-name> \
    --port 8770 \
    --peers "https://p2p.ringrift.ai" \
    --ringrift-path /root/ringrift \
    > /var/log/ringrift/p2p.log 2>&1 &

# Verify
pgrep -f p2p_orchestrator.py
```

### 7. Set Up Disk Cleanup Cron

```bash
# Deploy cleanup script
chmod +x ~/ringrift/ai-service/scripts/disk_cleanup.sh

# Add to crontab (runs every 6 hours)
(crontab -l 2>/dev/null | grep -v disk_cleanup; \
 echo "0 */6 * * * ~/ringrift/ai-service/scripts/disk_cleanup.sh >> ~/ringrift/ai-service/logs/disk_cleanup.log 2>&1") | crontab -
```

## Verification

### Check P2P Status

```bash
# From any node
curl -s https://p2p.ringrift.ai/status | python3 -c "
import json,sys
d = json.load(sys.stdin)
print(f'Leader: {d.get(\"leader_id\")}')
print(f'Voters: {d.get(\"voters_alive\")}/{len(d.get(\"voter_node_ids\",[]))}')
"
```

### Check Node is Visible

```bash
curl -s https://p2p.ringrift.ai/status | grep -o '"<node-name>"'
```

### Check Service Logs

```bash
# Systemd nodes
journalctl -u ringrift-p2p.service -n 50

# Vast nodes
tail -50 /var/log/ringrift/p2p.log
```

## Adding Node as Voter

Voters participate in leader election. Only add stable, reliable nodes.

1. Update `RINGRIFT_P2P_VOTERS` in `/etc/ringrift/node.conf` on ALL nodes
2. Add `p2p_voter: true` to the node entry in `distributed_hosts.yaml`
3. Restart P2P services on all nodes

## Troubleshooting

### Node Not Connecting

1. Check Tailscale: `tailscale status`
2. Check P2P port: `curl http://localhost:8770/health`
3. Check logs: `journalctl -u ringrift-p2p.service -n 100`

### P2P Service Restarting

1. Check timeout in node_resilience.py (should be 60s)
2. Check disk space: `df -h /`
3. Check memory: `free -h`

### High Disk Usage

1. Run cleanup manually: `~/ringrift/ai-service/scripts/disk_cleanup.sh`
2. Remove old checkpoints: `find ~/ringrift/ai-service/models -name "*_202*.pth" -mtime +1 -delete`

## Key Tailscale IPs

| Node           | Tailscale IP   |
| -------------- | -------------- |
| lambda-h100    | 100.78.101.123 |
| lambda-gh200-e | 100.88.176.74  |
| aws-staging    | 100.115.97.24  |
| aws-proxy      | 100.121.198.28 |

## Monitoring

- Prometheus: http://100.115.97.24:9090
- Grafana: http://100.115.97.24:3002 (admin/admin)
- Cluster Status: https://p2p.ringrift.ai/status

---

Last Updated: December 2025
