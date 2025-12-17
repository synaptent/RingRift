# Vast.ai Instance Lifecycle Management

This document covers the scripts and workflows for managing Vast.ai GPU instances in the RingRift AI cluster.

## Overview

Vast.ai instances are ephemeral GPU resources that require careful lifecycle management:

1. **Provisioning** - Spin up instances with correct configuration
2. **Integration** - Join instances to P2P cluster network
3. **Monitoring** - Track health and utilization
4. **Keepalive** - Prevent idle termination
5. **Scaling** - Auto-scale based on demand
6. **Termination** - Clean shutdown with data sync

## Scripts

### vast_p2p_sync.py (Primary Integration Script)

Synchronizes Vast.ai instances with the P2P network.

```bash
# Full sync: check instances, sync to P2P, start orchestrators
python scripts/vast_p2p_sync.py --full

# Check instance status only
python scripts/vast_p2p_sync.py --check

# Sync and update distributed_hosts.yaml
python scripts/vast_p2p_sync.py --sync

# Start P2P orchestrators on all instances
python scripts/vast_p2p_sync.py --start-p2p

# Provision new instances
python scripts/vast_p2p_sync.py --provision --gpu-type RTX_4090 --count 2
```

**Features:**

- Instance status tracking via Vast.ai CLI
- Node unretiring (restart terminated instances)
- Auto-updates `config/distributed_hosts.yaml`
- P2P orchestrator startup coordination
- SSH key management

### vast_keepalive.py (Keepalive Manager)

Prevents idle instance termination by maintaining heartbeats. Designed to run via cron every 15-30 minutes.

```bash
# Check all instances status
python scripts/vast_keepalive.py --status

# Send keepalive ping to all instances
python scripts/vast_keepalive.py --keepalive

# Restart stopped instances
python scripts/vast_keepalive.py --restart-stopped

# Full automation cycle (status + keepalive + restart)
python scripts/vast_keepalive.py --auto

# Install cron job locally (runs every 15 minutes)
python scripts/vast_keepalive.py --install-cron
```

**Features:**

- Periodic heartbeat to prevent idle termination
- Health check monitoring via SSH
- Auto-restart of stopped instances via Vast.ai CLI
- P2P network connectivity maintenance
- Code sync and worker restart on unhealthy instances
- Thread pool for parallel SSH operations

**Cron Setup:**

```bash
# Recommended: run every 15-30 minutes
*/15 * * * * cd ~/ringrift/ai-service && python scripts/vast_keepalive.py --auto >> /tmp/keepalive.log 2>&1
```

### vast_autoscaler.py (Demand-Based Scaling)

Automatically scales Vast.ai instances based on workload demand.

```bash
# Run autoscaler check (designed for cron)
python scripts/vast_autoscaler.py --check

# Scale up manually
python scripts/vast_autoscaler.py --scale-up --count 2

# Scale down idle instances
python scripts/vast_autoscaler.py --scale-down

# View current scaling policy
python scripts/vast_autoscaler.py --policy
```

**Features:**

- Queue depth monitoring (selfplay backlog)
- Budget constraints (daily/monthly limits)
- Time-of-day policies (cheaper night hours)
- Autoscaler group management
- Cooldown periods to prevent thrashing

**Cron Setup:**

```bash
# Check every 10-15 minutes
*/10 * * * * cd ~/ringrift/ai-service && python scripts/vast_autoscaler.py --check >> /tmp/autoscaler.log 2>&1
```

### vast_lifecycle.py (Instance Lifecycle)

Manages the complete lifecycle of Vast.ai instances.

```bash
# Start managed jobs on all instances
python scripts/vast_lifecycle.py --start-jobs

# Health check all instances
python scripts/vast_lifecycle.py --health-check

# Graceful shutdown with data sync
python scripts/vast_lifecycle.py --shutdown --sync-first

# List all managed instances
python scripts/vast_lifecycle.py --list
```

**Features:**

- Health checks with auto-restart
- Worker restart logic on failure
- Idle detection and cleanup
- Pre-termination data sync
- GPU-to-workload mapping

### vast_p2p_manager.py (P2P Network Management)

Manages P2P network topology for Vast.ai instances.

```bash
# View P2P network status
python scripts/vast_p2p_manager.py --status

# Force peer discovery
python scripts/vast_p2p_manager.py --discover

# Restart P2P on specific instance
python scripts/vast_p2p_manager.py --restart --instance vast-12345
```

## Configuration

### distributed_hosts.yaml

The primary configuration file for distributed hosts including Vast.ai instances.

```yaml
vast_instances:
  - id: vast-12345
    hostname: vast-12345.vast.ai
    tailscale_ip: 100.x.x.x
    gpu_type: RTX_4090
    gpu_count: 1
    status: active
    last_seen: 2025-12-17T10:00:00Z
```

### Environment Variables

| Variable             | Description          | Default               |
| -------------------- | -------------------- | --------------------- |
| `VAST_API_KEY`       | Vast.ai API key      | Required              |
| `VAST_SSH_KEY`       | Path to SSH key      | `~/.ssh/vast_rsa`     |
| `VAST_DEFAULT_IMAGE` | Default Docker image | `pytorch/pytorch:2.0` |
| `VAST_MAX_PRICE`     | Maximum $/hr to pay  | `0.50`                |
| `VAST_AUTOSCALE_MIN` | Minimum instances    | `2`                   |
| `VAST_AUTOSCALE_MAX` | Maximum instances    | `10`                  |

## Workflows

### New Instance Setup

1. Provision instance via `vast_p2p_sync.py --provision`
2. Instance boots and runs setup script
3. `vast_p2p_sync.py --sync` adds to distributed_hosts.yaml
4. `vast_p2p_sync.py --start-p2p` starts P2P orchestrator
5. Instance joins cluster and begins accepting work

### Instance Recovery

When an instance becomes unhealthy:

1. `vast_lifecycle.py --health-check` detects failure
2. Attempts restart via Vast.ai API
3. If restart fails, instance is marked for replacement
4. `vast_autoscaler.py` provisions replacement
5. Old instance data synced before termination

### Data Sync Before Termination

```bash
# Sync all data from instance before shutdown
python scripts/vast_lifecycle.py --shutdown --instance vast-12345 --sync-first
```

This ensures:

- All selfplay games are synced to coordinator
- Model checkpoints are backed up
- Elo database is synchronized
- No data loss on termination

## Monitoring

### Health Metrics

- GPU utilization (target: >60%)
- Memory usage (warning: >80%)
- Network connectivity (P2P mesh status)
- Worker process count
- Games generated per hour

### Alerts

Configure alerts in `config/notification_hooks.yaml`:

```yaml
vast_alerts:
  instance_unhealthy:
    webhook: ${SLACK_WEBHOOK_URL}
    threshold: 3 # failures before alert
  cost_warning:
    webhook: ${SLACK_WEBHOOK_URL}
    daily_threshold: 50.00 # USD
```

## Best Practices

1. **Always sync data before termination** - Use `--sync-first` flag
2. **Set budget limits** - Configure daily/monthly spending caps
3. **Use autoscaler cooldowns** - Prevent rapid scale up/down
4. **Monitor GPU utilization** - Idle instances waste money
5. **Keep keepalive daemon running** - Prevents unexpected termination
6. **Use spot instances wisely** - They can be preempted

## Troubleshooting

### Instance Won't Join P2P

1. Check Tailscale connectivity: `tailscale status`
2. Verify SSH access: `ssh vast-12345`
3. Check P2P orchestrator logs: `ssh vast-12345 'tail -100 /tmp/p2p_orchestrator.log'`
4. Force restart: `python scripts/vast_p2p_manager.py --restart --instance vast-12345`

### High Costs

1. Check for idle instances: `python scripts/vast_lifecycle.py --list`
2. Review autoscaler policy: `python scripts/vast_autoscaler.py --policy`
3. Terminate idle instances: `python scripts/vast_autoscaler.py --scale-down`

### Data Not Syncing

1. Check sync status: `python scripts/unified_data_sync.py --status`
2. Verify network: `ping <coordinator_ip>`
3. Manual sync: `python scripts/unified_data_sync.py --source vast --dest local --force`
