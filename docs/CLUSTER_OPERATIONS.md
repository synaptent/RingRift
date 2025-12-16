# RingRift Cluster Operations Guide

## Overview

This document covers the operational scripts and monitoring tools for the RingRift distributed training cluster.

## Architecture

```
Primary Hub: lambda-h100 (209.20.157.81 / 100.78.101.123)
Backup Hub:  lambda-gh200-e (100.88.176.74)
Tunnel Ports: 8771-8782 (primary), 8877-8881 (backup)
```

## Monitoring Scripts

### 1. Cluster Alerting (`scripts/cluster_alerting.sh`)
- **Purpose**: Sends alerts to Slack/Discord when issues detected
- **Schedule**: Every 5 minutes via cron
- **Configuration**: Set `ALERT_WEBHOOK_URL` in `.env`
- **Manual run**: `./scripts/cluster_alerting.sh --test`

Alerts on:
- P2P service down
- Low peer count (<50%)
- High disk usage (>85%)
- High memory usage (>90%)
- No leader elected
- Tunnels down

### 2. Tunnel Watchdog (`scripts/tunnel_watchdog.sh`)
- **Purpose**: Auto-recovers failed Lambda node tunnels
- **Schedule**: Every 10 minutes via cron
- **Manual run**: `./scripts/tunnel_watchdog.sh`

### 3. P2P Tuning (`scripts/p2p_tuning.sh`)
- **Purpose**: Monitors and optimizes P2P network stability
- **Schedule**: Every 15 minutes via cron
- **Actions**: Triggers anti-entropy sync when peer ratio is low

### 4. Prometheus Metrics (`scripts/prometheus_metrics.sh`)
- **Purpose**: Exports cluster metrics in Prometheus format
- **Usage**: `./scripts/prometheus_metrics.sh`
- **Metrics exported**:
  - `ringrift_active_peers`
  - `ringrift_selfplay_jobs`
  - `ringrift_training_jobs`
  - `ringrift_disk_percent`
  - `ringrift_tunnels_up/down`

### 5. Training Optimizer (`scripts/training_optimizer.sh`)
- **Purpose**: Monitors training backlog and node availability
- **Schedule**: Every 30 minutes via cron

### 6. Disk Cleanup (`scripts/disk_cleanup.sh`)
- **Purpose**: Automated disk space management
- **Schedule**: Daily at 3am via cron
- **Actions**:
  - Removes corrupted files
  - Cleans old logs (>7 days)
  - Compresses old databases (>30 days)

## Vast.ai Node Management

### SSH Access
```bash
# Via Vast gateway
ssh -p <port> root@ssh<N>.vast.ai

# Example instances:
# vast-3070:    ssh -p 14364 root@ssh5.vast.ai
# vast-2060s:   ssh -p 14370 root@ssh2.vast.ai
# vast-5090:    ssh -p 14398 root@ssh7.vast.ai
```

### Manage Instances
```bash
# List instances
vastai show instances

# Reboot instance
vastai reboot instance <ID>

# Start/stop
vastai start instance <ID>
vastai stop instance <ID>
```

### Setup New Vast Instance
1. SSH into the instance
2. Clone RingRift: `git clone https://github.com/yourusername/ringrift.git`
3. Install deps: `pip install -r ai-service/requirements.txt`
4. Start P2P: `python -m scripts.p2p_orchestrator --node-id <name> --seeds http://209.20.157.81:8770`
5. Setup autossh tunnel to primary hub

## Tunnel Management

### Port Assignments (Primary Hub)
| Port | Node |
|------|------|
| 8771 | lambda-gh200-a |
| 8772 | lambda-gh200-b |
| 8773 | lambda-gh200-c |
| 8774 | lambda-gh200-d |
| 8775 | lambda-gh200-e |
| 8776 | lambda-gh200-i |
| 8777 | vast-3070 |
| 8778 | vast-2060s |
| 8779 | lambda-gh200-h |
| 8780 | vast-262969f8 |
| 8781 | lambda-gh200-g |
| 8782 | vast-3060ti |

### Check Tunnel Status
```bash
# Quick check
for port in 877{1..9} 878{0..2}; do
    nc -z -w1 localhost $port && echo "$port: UP" || echo "$port: DOWN"
done
```

## Cron Jobs Summary

```cron
*/5 * * * *  cluster_alerting.sh      # Alerting
*/10 * * * * tunnel_watchdog.sh       # Tunnel recovery
*/15 * * * * p2p_tuning.sh            # P2P optimization
*/30 * * * * training_optimizer.sh    # Training monitoring
0 3 * * *    disk_cleanup.sh          # Daily cleanup
```

## Troubleshooting

### P2P Service Not Responding
```bash
sudo systemctl restart ringrift-p2p
journalctl -u ringrift-p2p -f
```

### Low Peer Count
1. Check tunnel status
2. Run `./scripts/p2p_tuning.sh`
3. Restart P2P on affected nodes

### High Disk Usage
```bash
# Manual cleanup
./scripts/disk_cleanup.sh

# Check largest files
du -sh /home/ubuntu/ringrift/ai-service/data/*
```

### Vast Node Unreachable
1. Check via `vastai show instances`
2. Reboot: `vastai reboot instance <ID>`
3. Re-setup P2P and tunnels after reboot
