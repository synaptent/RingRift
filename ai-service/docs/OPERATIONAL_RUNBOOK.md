# RingRift Cluster Operational Runbook

## Quick Reference

### Common Commands

```bash
# Check cluster status
python scripts/vast_p2p_sync.py --check

# Check autoscaler status
python scripts/vast_autoscaler.py --status

# Run health checks
python scripts/health_alerting.py --check

# Sync code to all nodes
python scripts/vast_p2p_sync.py --sync-code

# Monitor cluster (live)
./scripts/cluster_monitoring.sh --loop
```

### Emergency Commands

```bash
# Stop all Vast instances
vastai show instances --raw | python3 -c "import json,sys; [print(i['id']) for i in json.load(sys.stdin) if i.get('actual_status')=='running']" | xargs -I{} vastai stop instance {}

# Restart P2P on all nodes
python scripts/vast_p2p_sync.py --full

# Force keepalive
python scripts/vast_keepalive.py --keepalive
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      P2P Mesh Network (Tailscale)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  mac-studio  │  │ lambda-h100  │  │ lambda-gh200 │  x10         │
│  │   (Leader)   │  │   (Voter)    │  │   (Voter)    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│         │                  │                  │                     │
│         └──────────────────┼──────────────────┘                     │
│                            │                                        │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                 Vast.ai GPU Instances                     │      │
│  │   RTX 3070 x4  |  RTX 5090 x2  |  A40 x1  |  ...         │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component        | Location       | Port | Purpose            |
| ---------------- | -------------- | ---- | ------------------ |
| P2P Orchestrator | All nodes      | 8770 | Job coordination   |
| Model Server     | Training nodes | 8766 | Model distribution |
| aria2 RPC        | Vast nodes     | 6800 | Parallel downloads |
| Tailscale SOCKS  | Vast nodes     | 1055 | Mesh connectivity  |

---

## Cron Jobs

The following cron jobs run automatically:

| Schedule        | Script                       | Purpose                  |
| --------------- | ---------------------------- | ------------------------ |
| _/15 _ \* \* \* | vast_keepalive.py --auto     | Prevent idle termination |
| _/30 _ \* \* \* | vast_orchestrator_cron.sh    | Full orchestration cycle |
| _/15 _ \* \* \* | cluster_automation.py --full | Cluster automation       |

### Manual Cron Installation

```bash
# View current cron
crontab -l

# Edit cron
crontab -e

# Add these lines:
*/15 * * * * cd /path/to/ai-service && python scripts/vast_keepalive.py --auto >> logs/vast_keepalive.log 2>&1
*/30 * * * * /path/to/ai-service/scripts/vast_orchestrator_cron.sh
```

---

## Common Scenarios

### Scenario 1: Vast Instance Not Joining P2P

**Symptoms:**

- Instance shows as running in `vastai show instances`
- Node doesn't appear in P2P network
- Selfplay not running

**Resolution:**

```bash
# 1. Check SSH connectivity
ssh -p PORT root@ssh5.vast.ai "echo OK"

# 2. Check P2P status on instance
ssh -p PORT root@ssh5.vast.ai "curl -s localhost:8770/health"

# 3. Manually start P2P
ssh -p PORT root@ssh5.vast.ai "cd ~/ringrift/ai-service && pkill -f p2p_orchestrator; nohup python scripts/p2p_orchestrator.py --node-id vast-INSTANCE_ID --port 8770 --peers 100.107.168.125:8770 > logs/p2p.log 2>&1 &"

# 4. Or use the sync script
python scripts/vast_p2p_sync.py --full
```

### Scenario 2: High Costs / Too Many Instances

**Symptoms:**

- Hourly cost > $10
- Many idle instances

**Resolution:**

```bash
# 1. Check current costs
vastai show instances --raw | python3 -c "import json,sys; d=json.load(sys.stdin); r=[i for i in d if i.get('actual_status')=='running']; print(f'Hourly: \${sum(i.get(\"dph_total\",0) for i in r):.2f}')"

# 2. Stop idle instances (dry-run first)
python scripts/vast_autoscaler.py --dry-run

# 3. Actually scale down
python scripts/vast_autoscaler.py --scale

# 4. Manually stop expensive instances
vastai stop instance INSTANCE_ID
```

### Scenario 3: Model Not Syncing

**Symptoms:**

- New model trained but not on other nodes
- Old model being used for selfplay

**Resolution:**

```bash
# 1. Check model status
python scripts/model_sync_aria2.py --status

# 2. Sync to all nodes
python scripts/model_sync_aria2.py --sync-to-all

# 3. Manual rsync fallback
rsync -avz --progress data/models/latest.pth ubuntu@100.88.176.74:~/ringrift/ai-service/data/models/
```

### Scenario 4: P2P Network Partition

**Symptoms:**

- Some nodes can't see others
- Leader election stuck
- Multiple leaders

**Resolution:**

```bash
# 1. Check network status
python scripts/vast_p2p_sync.py --check

# 2. Verify Tailscale connectivity
tailscale status

# 3. Restart P2P on all nodes
python scripts/vast_p2p_setup.py --deploy-to-vast --components p2p

# 4. Check for retired nodes and unretire
python scripts/vast_p2p_sync.py --sync
```

### Scenario 5: Disk Full on Node

**Symptoms:**

- Workers crashing
- "No space left on device" errors

**Resolution:**

```bash
# 1. Check disk usage remotely
ssh ubuntu@NODE_IP "df -h"

# 2. Clean up old data
ssh ubuntu@NODE_IP "cd ~/ringrift/ai-service && find data/games -name '*.db' -mtime +7 -delete"

# 3. Clean old models
ssh ubuntu@NODE_IP "cd ~/ringrift/ai-service && ls -t data/models/*.pth | tail -n +10 | xargs rm -f"
```

---

## Monitoring

### Health Checks

```bash
# Run all health checks
python scripts/health_alerting.py --check

# Check without sending alerts
python scripts/health_alerting.py --check --no-alerts
```

### Key Metrics to Watch

| Metric         | Normal | Warning | Critical |
| -------------- | ------ | ------- | -------- |
| P2P Peers      | > 15   | 10-15   | < 10     |
| Vast Instances | 10-15  | 5-10    | < 5      |
| Selfplay Jobs  | > 500  | 100-500 | < 100    |
| Hourly Cost    | < $8   | $8-12   | > $12    |
| Idle Instances | < 3    | 3-5     | > 5      |

### Grafana Dashboard

Import the dashboard from: `config/grafana/cluster_dashboard.json`

---

## Autoscaling

### Check Autoscaler Status

```bash
python scripts/vast_autoscaler.py --status
```

### Manual Scale Up

```bash
# Provision 2 new instances
python scripts/vast_p2p_sync.py --provision 2 --max-hourly 0.50
```

### Manual Scale Down

```bash
# Preview scale-down
python scripts/vast_autoscaler.py --dry-run

# Execute scale-down
python scripts/vast_autoscaler.py --scale
```

### Autoscaler Groups

```bash
# List groups
python scripts/vast_autoscaler.py --list-groups

# Create a group
python scripts/vast_autoscaler.py --create-group selfplay-rtx3070 --gpu "RTX 3070" --target 5 --max-price 0.08
```

---

## Alerting

### Configure Webhooks

Set environment variables:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
export PAGERDUTY_ROUTING_KEY="..."
```

Or add to `config/notification_hooks.yaml`:

```yaml
hooks:
  - type: slack
    enabled: true
    webhook_url: 'https://hooks.slack.com/services/...'
```

### Test Alerts

```bash
python scripts/health_alerting.py --alert-test
```

---

## Rollback Procedures

### Model Rollback

If a newly promoted model performs poorly:

```bash
# Check current model status
python scripts/elo_reconciliation_cli.py model-status

# Trigger manual rollback
python scripts/elo_reconciliation_cli.py rollback --board-type hexagonal --num-players 2

# Check rollback history
python scripts/elo_reconciliation_cli.py rollback-history
```

### Code Rollback

```bash
# On all nodes
git reset --hard HEAD~1

# Sync to cluster
python scripts/vast_p2p_sync.py --sync-code
```

---

## Contact & Escalation

For issues not covered in this runbook:

1. Check logs: `logs/vast_orchestrator_cron.log`, `logs/p2p_orchestrator.log`
2. Check P2P status: `curl http://localhost:8770/status`
3. Review recent changes: `git log --oneline -10`

---

## Appendix: Full Command Reference

### vast_p2p_sync.py

```
--check         Check status only
--sync          Sync and unretire active instances
--start-p2p     Start P2P on instances missing it
--full          Full sync (check + sync + start + config)
--update-config Update distributed_hosts.yaml
--provision N   Provision N new instances
--sync-code     Sync git code to all instances
```

### vast_autoscaler.py

```
--status        Show scaling status
--scale         Execute scaling decisions
--auto          Full auto-scaling cycle
--dry-run       Preview without changes
--list-groups   List autoscaler groups
--create-group  Create autoscaler group
```

### health_alerting.py

```
--check         Run health checks
--alert-test    Test alert delivery
--daemon        Run continuously
--no-alerts     Suppress alert sending
```

### model_sync_aria2.py

```
--status        Show sync status
--sync-to-all   Sync to all nodes
--sync-from     Pull from source
--serve         Start model server
```
