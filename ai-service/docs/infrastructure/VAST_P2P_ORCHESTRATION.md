# Vast.ai P2P Orchestration System

## Overview

This document describes the automated P2P orchestration system for Vast.ai GPU instances, integrating Tailscale mesh networking, aria2 parallel downloads, and optional Cloudflare tunnels for NAT traversal.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        P2P Mesh Network                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │ coordinator │    │ gpu-node-1  │    │ gpu-node-2  │             │
│  │ (Leader)    │◄──►│ (Voter)     │◄──►│ (Voter)     │             │
│  │ TS: 100.x   │    │ TS: 100.x   │    │ TS: 100.x   │             │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘             │
│         │                  │                  │                     │
│         └──────────────────┼──────────────────┘                     │
│                            │                                        │
│         ┌──────────────────┼──────────────────┐                     │
│         │                  │                  │                     │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐             │
│  │ cloud-vm-1  │    │ cloud-vm-2  │    │ cloud-vm-3  │  ... x N    │
│  │ P2P:8770    │    │ P2P:8770    │    │ P2P:8770    │             │
│  │ aria2:6800  │    │ aria2:6800  │    │ aria2:6800  │             │
│  │ data:8766   │    │ data:8766   │    │ data:8766   │             │
│  │ SOCKS:1055  │    │ SOCKS:1055  │    │ SOCKS:1055  │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. P2P Orchestrator (`scripts/p2p_orchestrator.py`)

- Runs on every node (port 8770)
- Leader election via Bully algorithm
- Job distribution and monitoring
- Heartbeat-based failure detection

### 2. Tailscale SOCKS5 Proxy

- Userspace networking for containers without CAP_NET_ADMIN
- SOCKS5 server on `localhost:1055`
- Enables mesh connectivity through NAT/firewalls
- P2P uses via `RINGRIFT_SOCKS_PROXY=socks5://localhost:1055`

### 3. aria2 Data Server

- RPC server on port 6800 for programmatic downloads
- HTTP data server on port 8766 for file serving
- 16 parallel connections per source
- Multi-source metalink support

### 4. Cloudflare Tunnels (Optional)

- Quick tunnels for NAT bypass
- No open ports required
- URLs change on restart (use named tunnels for persistence)
- For the public cluster entrypoint, use the named tunnel config in
  `config/cloudflared-config.yaml` with `scripts/setup_cloudflare_tunnel.sh`
  on the backbone nodes (not on Vast instances).

## Setup Scripts

### Primary Script: `scripts/vast_p2p_setup.py`

```bash
# Check status of all Vast instances
python scripts/vast_p2p_setup.py --check-status

# Deploy all components
python scripts/vast_p2p_setup.py --deploy-to-vast --components tailscale aria2 p2p

# Deploy specific components
python scripts/vast_p2p_setup.py --deploy-to-vast --components aria2

# Add Cloudflare tunnels (optional)
python scripts/vast_p2p_setup.py --deploy-to-vast --components cloudflare
```

### Lifecycle Manager: `scripts/vast_lifecycle.py`

```bash
# Health check with P2P status
python scripts/vast_lifecycle.py --check

# Deploy P2P orchestrator
python scripts/vast_lifecycle.py --deploy-p2p

# Update distributed_hosts.yaml
python scripts/vast_lifecycle.py --update-config

# Start jobs on idle instances
python scripts/vast_lifecycle.py --start-jobs

# Full automation cycle
python scripts/vast_lifecycle.py --auto
```

## Configuration

### distributed_hosts.yaml

Vast instances are auto-discovered and added to `config/distributed_hosts.yaml`:

```yaml
hosts:
  vast-INSTANCE_ID:
    ssh_host: sshN.vast.ai # From `vastai show instances`
    ssh_port: 12345 # From `vastai show instances`
    ssh_user: root
    ssh_key: ~/.ssh/id_cluster
    ringrift_path: ~/ringrift/ai-service
    memory_gb: 773
    cpus: 512
    gpu: 8x RTX 5090
    role: nn_training_primary
    status: ready
    vast_instance_id: 'INSTANCE_ID'
    tailscale_ip: 100.x.x.x # Tailscale IP if available
```

### GPU to Role Mapping

| GPU                       | VRAM    | Role                | Board Type |
| ------------------------- | ------- | ------------------- | ---------- |
| RTX 3070, 2060S, 3060 Ti  | ≤8GB    | gpu_selfplay        | hex8       |
| RTX 4060 Ti, 4080S, 5080  | 12-16GB | gpu_selfplay        | square8    |
| RTX 5070, 5090, A40, H100 | 24GB+   | nn_training_primary | hexagonal  |

## Using aria2 for Parallel Downloads

### Via aria2_transport.py

```python
from app.distributed.aria2_transport import Aria2Transport, Aria2Config

transport = Aria2Transport(Aria2Config(
    connections_per_server=16,
    split=16,
    max_concurrent_downloads=5,
))

# Sync from multiple sources (replace with your instance Tailscale IPs or hostnames)
result = await transport.sync_from_sources(
    sources=["http://node-1:8766", "http://node-2:8766"],
    local_dir=Path("data/models"),
    patterns=["*.pth"],
)
```

### Via aria2c CLI

```bash
# Download from multiple sources with 16 connections each
# Replace node-1, node-2 with your instance Tailscale IPs or hostnames
aria2c --max-connection-per-server=16 --split=16 \
    http://node-1:8766/models/latest.pth \
    http://node-2:8766/models/latest.pth
```

## Environment Variables

| Variable               | Description                    | Default       |
| ---------------------- | ------------------------------ | ------------- |
| `RINGRIFT_SOCKS_PROXY` | SOCKS5 proxy URL               | (none)        |
| `RINGRIFT_P2P_VERBOSE` | Enable verbose logging         | false         |
| `RINGRIFT_P2P_VOTERS`  | Comma-separated voter node IDs | (from config) |

## Monitoring

### Check P2P Health

```bash
# From any node
curl http://localhost:8770/health

# Get cluster status
curl http://localhost:8770/cluster/status
```

### Check aria2 Status

```bash
# Via JSON-RPC
curl http://localhost:6800/jsonrpc \
    -d '{"jsonrpc":"2.0","method":"aria2.getGlobalStat","id":1}'
```

## Troubleshooting

### P2P Won't Start

1. Check Python imports: `python -c "import scripts.p2p.types"`
2. Check venv: `source venv/bin/activate`
3. Check logs: `cat logs/p2p_orchestrator.log`

### Tailscale SOCKS Not Working

1. Check tailscaled is running: `pgrep tailscaled`
2. Test SOCKS: `curl --socks5 localhost:1055 http://COORDINATOR_IP:8770/health`
3. Check auth: `tailscale status`

### aria2 Not Responding

1. Check process: `pgrep aria2c`
2. Test RPC: `curl http://localhost:6800/jsonrpc -d '{"jsonrpc":"2.0","method":"aria2.getVersion","id":1}'`
3. Check data server: `curl http://localhost:8766/`

## Keepalive Manager

The keepalive manager prevents idle termination and maintains instance health.

**Script:** `scripts/vast_keepalive.py`

### Usage

```bash
# Check status of all instances
python scripts/vast_keepalive.py --status

# Send keepalive to all instances
python scripts/vast_keepalive.py --keepalive

# Restart stopped instances
python scripts/vast_keepalive.py --restart-stopped

# Full automation cycle (status + keepalive + restart + fix unhealthy)
python scripts/vast_keepalive.py --auto

# Install cron job (runs every 15 minutes)
python scripts/vast_keepalive.py --install-cron
```

### What It Does

1. **Status Check:** Monitors instance reachability, Tailscale IPs, P2P status
2. **Keepalive:** Sends commands to prevent idle termination
3. **Worker Management:** Restarts stopped selfplay workers
4. **Code Sync:** Syncs code on unhealthy instances
5. **P2P Recovery:** Ensures P2P orchestrator is running

### Cron Schedule

Recommended: Every 15-30 minutes

```bash
*/15 * * * * cd ~/ringrift/ai-service && python scripts/vast_keepalive.py --auto >> logs/vast_keepalive_cron.log 2>&1
```

### Logs

- Main log: `logs/vast_keepalive.log`
- Cron log: `logs/vast_keepalive_cron.log`

---

## P2P Sync System

Synchronizes Vast instance state with P2P network membership.

**Script:** `scripts/vast_p2p_sync.py`

### Usage

```bash
# Check status only
python scripts/vast_p2p_sync.py --check

# Sync and unretire active instances in P2P network
python scripts/vast_p2p_sync.py --sync

# Start P2P orchestrator on instances missing it
python scripts/vast_p2p_sync.py --start-p2p

# Full sync (check + sync + start)
python scripts/vast_p2p_sync.py --full

# Update distributed_hosts.yaml with current IPs
python scripts/vast_p2p_sync.py --update-config

# Provision N new instances
python scripts/vast_p2p_sync.py --provision 3
```

### What It Does

1. **Instance Discovery:** Gets active instances from vastai CLI
2. **P2P Comparison:** Compares with P2P network retired nodes
3. **Unretire:** Calls `/admin/unretire` for nodes matching active instances
4. **P2P Startup:** Starts P2P orchestrator on instances without it
5. **Config Update:** Updates `config/distributed_hosts.yaml`
6. **Provisioning:** Auto-provisions instances based on GPU preferences

### Admin Endpoints Used

```bash
# Unretire a node (replace INSTANCE_ID with actual instance ID)
curl -X POST http://localhost:8770/admin/unretire/vast-INSTANCE_ID

# Get retired nodes
curl http://localhost:8770/cluster/retired
```

### GPU Role Mapping

The sync system automatically assigns roles based on GPU type:

| GPU                   | Role                  | Use Case                         |
| --------------------- | --------------------- | -------------------------------- |
| RTX 3070, 3060, 2060S | `gpu_selfplay`        | Self-play data generation        |
| RTX 4060 Ti, 4080S    | `gpu_selfplay`        | Self-play with larger boards     |
| RTX 5070, 5080, 5090  | `nn_training_primary` | Neural network training          |
| A10, A40, H100        | `nn_training_primary` | Training + large batch inference |

### Auto-Provisioning

Provision instances with preferred GPU types:

```bash
# Provision 3 instances with best available GPUs
python scripts/vast_p2p_sync.py --provision 3
```

GPU preference order:

1. RTX 3070 (max $0.08/hr)
2. RTX 3060 (max $0.06/hr)
3. RTX 4060 Ti (max $0.12/hr)
4. RTX 2080 Ti (max $0.10/hr)

### Logs

- Main log: `logs/vast_p2p_sync.log`

---

## Orchestrator Cron Integration

The keepalive and P2P sync are integrated into a unified cron system.

### Recommended Cron Setup

```bash
# Edit crontab
crontab -e

# Add these entries:
# Keepalive every 15 minutes
*/15 * * * * cd ~/ringrift/ai-service && python scripts/vast_keepalive.py --auto >> logs/vast_keepalive_cron.log 2>&1

# P2P sync every 30 minutes
*/30 * * * * cd ~/ringrift/ai-service && python scripts/vast_p2p_sync.py --full >> logs/vast_p2p_sync_cron.log 2>&1

# Config update hourly
0 * * * * cd ~/ringrift/ai-service && python scripts/vast_p2p_sync.py --update-config >> logs/vast_config_cron.log 2>&1
```

### Install All Cron Jobs

```bash
python scripts/vast_keepalive.py --install-cron
```

---

## Related Files

- `scripts/vast_p2p_setup.py` - Unified setup for SOCKS, aria2, P2P
- `scripts/vast_lifecycle.py` - Instance lifecycle management
- `scripts/vast_keepalive.py` - Keepalive manager (prevents idle termination)
- `scripts/vast_p2p_sync.py` - P2P network synchronization
- `scripts/p2p_orchestrator.py` - Main P2P orchestrator
- `app/distributed/aria2_transport.py` - aria2 transport layer
- `scripts/setup_cloudflare_tunnel.sh` - Cloudflare tunnel setup (cluster entrypoint)
- `config/cloudflared-config.yaml` - Cloudflare tunnel ingress rules
- `config/distributed_hosts.yaml` - Host configuration
