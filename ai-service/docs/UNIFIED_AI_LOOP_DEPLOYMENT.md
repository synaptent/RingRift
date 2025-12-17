# Unified AI Self-Improvement Loop - Deployment Guide

This guide covers deploying and operating the unified AI self-improvement loop across the RingRift cluster.

## Overview

The unified AI loop is a single daemon that coordinates all aspects of the AI improvement cycle:

- **Streaming Data Collection** - 30-second incremental sync from all hosts (optimized)
- **Shadow Tournaments** - 5-minute lightweight Elo evaluation (15 games/config)
- **Training Scheduler** - Auto-trigger training when 300+ new games accumulated
- **Model Promoter** - Auto-deploy models that exceed 20+ Elo threshold (15min cooldown)
- **Adaptive Curriculum** - Elo-weighted training focus on underperforming configs
- **Regression Gate** - Block promotions that fail regression tests

### Performance Targets

| Metric                  | Before          | After (Optimized)          |
| ----------------------- | --------------- | -------------------------- |
| Data sync latency       | 30 min          | ~30 sec                    |
| Elo evaluation feedback | 6 hours         | 5 min                      |
| Training trigger        | Manual          | Auto (300 games threshold) |
| Model deployment        | Manual          | Auto (20+ Elo, 15min cool) |
| Model reload in workers | Process restart | Hot reload                 |

## Prerequisites

### Required on Primary Host (lambda_h100)

```bash
# Python dependencies
pip install prometheus_client pyyaml

# Verify installation
python3 -c "from prometheus_client import Counter; print('OK')"
```

### Required on All Hosts

- Python 3.10+
- SSH access configured (key-based auth)
- Tailscale connected (for Tailscale-addressed hosts)
- `rsync` installed

### Network Requirements

| Port | Service            | Access       |
| ---- | ------------------ | ------------ |
| 22   | SSH                | All hosts    |
| 9090 | Prometheus metrics | Primary host |

## Quick Start

### 1. Deploy to All Hosts

From the local development machine:

```bash
cd ai-service
./scripts/deploy_unified_loop.sh
```

### 2. Start the Unified Loop

```bash
ssh ubuntu@209.20.157.81 'cd ~/ringrift/ai-service && \
  nohup python3 scripts/unified_ai_loop.py --foreground -v \
  > logs/unified_loop/daemon.log 2>&1 &'
```

### 3. Verify It's Running

```bash
ssh ubuntu@209.20.157.81 'cd ~/ringrift/ai-service && \
  python3 scripts/unified_ai_loop.py --status'
```

Expected output:

```
Unified AI Loop Status:
  Started: 2025-12-14T21:00:31
  Last cycle: 2025-12-14T21:05:00
  Data syncs: 5
  Training runs: 0
  Evaluations: 3
  Promotions: 0
```

## Deployment Script Options

The `deploy_unified_loop.sh` script supports several modes:

```bash
# Deploy to all hosts (default)
./scripts/deploy_unified_loop.sh

# Deploy only to primary coordinator
./scripts/deploy_unified_loop.sh --primary

# Deploy only to selfplay hosts
./scripts/deploy_unified_loop.sh --selfplay

# Install systemd services for auto-start on boot
./scripts/deploy_unified_loop.sh --install-systemd

# Restart the unified loop after deployment
./scripts/deploy_unified_loop.sh --restart

# Dry run - show what would be done
./scripts/deploy_unified_loop.sh --dry-run

# Disable hot reload configuration
./scripts/deploy_unified_loop.sh --no-hot-reload
```

## Configuration

### Main Configuration File

`config/unified_loop.yaml` controls all aspects of the loop:

```yaml
# Data ingestion from remote hosts
data_ingestion:
  poll_interval_seconds: 60 # How often to check for new games
  sync_method: 'incremental' # Use rsync incremental mode
  min_games_per_sync: 5 # Minimum games before syncing

# Training triggers
training:
  trigger_threshold_games: 500 # Start training after this many new games
  min_interval_seconds: 1200 # At least 20 min between training runs

# Evaluation settings
evaluation:
  shadow_interval_seconds: 900 # Shadow eval every 15 minutes
  shadow_games_per_config: 15 # Games per shadow tournament
  full_tournament_interval_seconds: 3600 # Full eval every hour

# Auto-promotion
promotion:
  auto_promote: true
  elo_threshold: 25 # Must beat current best by 25 Elo
  min_games: 50 # Minimum games for reliability
  regression_test: true # Run regression gate before promotion

# Adaptive curriculum
curriculum:
  adaptive: true
  rebalance_interval_seconds: 3600
  max_weight_multiplier: 1.5 # Max boost for underperforming configs
```

### Host Configuration

`config/remote_hosts.yaml` defines all cluster hosts:

```yaml
standard_hosts:
  lambda_h100:
    ssh_host: '209.20.157.81'
    ssh_user: 'ubuntu'
    role: 'training,hp_tuning,tournament'
    has_gpu: true
    gpu_type: 'H100'

  gh200_a:
    ssh_host: '192.222.51.29'
    ssh_user: 'ubuntu'
    role: 'selfplay'
    has_gpu: true
    gpu_type: 'GH200'
```

## Systemd Service (Production)

For production deployments, use systemd for automatic restart and boot-time startup.

### Install Services

```bash
./scripts/deploy_unified_loop.sh --install-systemd
```

Or manually:

```bash
# On primary host
sudo cp config/systemd/ringrift-ai-loop.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ringrift-ai-loop
sudo systemctl start ringrift-ai-loop
```

### Manage Service

```bash
# Start/stop/restart
sudo systemctl start ringrift-ai-loop
sudo systemctl stop ringrift-ai-loop
sudo systemctl restart ringrift-ai-loop

# Check status
sudo systemctl status ringrift-ai-loop

# View logs
sudo journalctl -u ringrift-ai-loop -f
```

### Selfplay Workers (Optional)

Selfplay workers can also be managed via systemd:

```bash
# Install on selfplay hosts
sudo cp config/systemd/ringrift-selfplay@.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start worker for specific config
sudo systemctl start ringrift-selfplay@square8_2p
sudo systemctl start ringrift-selfplay@square19_3p
```

## Monitoring

### Prometheus Metrics

The unified loop exposes metrics at `http://<primary-host>:9090/metrics`.

Key metrics:

| Metric                            | Description                                  |
| --------------------------------- | -------------------------------------------- |
| `ringrift_games_synced_total`     | Total games synced by host                   |
| `ringrift_games_pending_training` | Games awaiting training by config            |
| `ringrift_training_runs_total`    | Training runs by config and status           |
| `ringrift_current_elo`            | Current Elo rating by config                 |
| `ringrift_elo_trend`              | Elo improvement trend (positive = improving) |
| `ringrift_promotions_total`       | Model promotions by config                   |
| `ringrift_hosts_active`           | Number of reachable hosts                    |

### Grafana Dashboards

**AI Self-Improvement Loop Dashboard** (new):

1. Open Grafana → Dashboards → Import
2. Upload `ai-service/config/monitoring/grafana-dashboard.json`
3. Select your Prometheus data source
4. Dashboard will appear as "RingRift AI Self-Improvement Loop"

Dashboard panels:

- Uptime and active hosts
- Games synced per hour
- Elo ratings by configuration

**AI Cluster Dashboard** (existing at `cluster.ringrift.ai`):

The main cluster dashboard at `/monitoring/grafana/dashboards/ai-cluster.json` requires:

- Infinity datasource plugin installed in Grafana
- Orchestrator host variable set to `209.20.157.81` (lambda_h100)
- P2P Orchestrator running on port 8770

**Fixing "No data" in Elo Leaderboard or Node Status Table:**

```bash
# 1. Verify orchestrator is running
ssh ubuntu@209.20.157.81 'curl -s http://localhost:8770/elo/table | head -5'

# 2. If not running, start it
ssh ubuntu@209.20.157.81 'cd ~/ringrift/ai-service && \
  nohup python scripts/p2p_orchestrator.py --port 8770 > logs/orchestrator.log 2>&1 &'
```

Then in Grafana:

1. Go to Dashboard Settings → Variables
2. Set `orchestrator_host` to `209.20.157.81`
3. Save and refresh the dashboard

### Alerting

Configure alerts in Prometheus using `config/monitoring/alerting-rules.yaml`:

**Critical Alerts:**

- `AILoopDown` - Loop process is down for 5+ minutes
- `NoGamesSynced` - No games synced in 30 minutes
- `AllHostsFailed` - All hosts unreachable

**High Severity:**

- `TrainingStuck` - Training running for 2+ hours
- `HighSyncErrorRate` - Sync errors exceeding threshold
- `GamesPendingThresholdExceeded` - 5000+ games pending training

**Warning:**

- `NoPromotionsIn24h` - Training but no improvements
- `EloDecline` - Elo trending negative
- `LowActiveHosts` - Fewer than 5 hosts active

### Quick Health Check

```bash
# Check if loop is running
curl -s http://209.20.157.81:9090/health

# Get key metrics
curl -s http://209.20.157.81:9090/metrics | grep -E 'ringrift_(uptime|hosts_active|games_synced)'
```

## Hot Model Reload

Selfplay workers can reload models without restart when new models are promoted.

### How It Works

1. Unified loop promotes a new model
2. Model file is synced to all hosts
3. Workers detect file change via mtime check
4. Workers reload the model on next game start

### Enable Hot Reload

Hot reload is enabled during deployment. To manually enable:

```bash
# On each selfplay host
mkdir -p ~/ringrift/ai-service/data/model_updates
touch ~/ringrift/ai-service/data/model_updates/.hot_reload_enabled
```

### Verify Hot Reload

Check worker logs for reload messages:

```bash
grep "Reloading model" ~/ringrift/ai-service/logs/selfplay/*.log
```

## Regression Gate

The regression gate prevents promoting models that regress on key metrics.

### Tests Performed

1. **Model file validation** - File exists and has valid size
2. **Inference performance** - Latency under 50ms
3. **No illegal moves** - Model never makes illegal moves
4. **Game completion** - Games complete without hanging
5. **Win rate vs random** - >90% win rate
6. **Win rate vs heuristic** - >55% win rate

### Manual Testing

```bash
python scripts/regression_gate.py \
  --model models/square8_2p/best.pt \
  --config square8_2p \
  --verbose
```

### Configure Thresholds

Edit `config/unified_loop.yaml`:

```yaml
regression:
  hard_block: true # Block promotion on failure
  timeout_seconds: 600 # Max test duration
```

## Troubleshooting

### Loop Not Starting

```bash
# Check for Python errors
ssh ubuntu@209.20.157.81 'cd ~/ringrift/ai-service && \
  python3 scripts/unified_ai_loop.py --foreground -v'

# Common issues:
# - Missing prometheus_client: pip install prometheus_client
# - Port 9090 in use: Change metrics_port in config
# - YAML parse error: Validate config/unified_loop.yaml
```

### No Data Being Synced

```bash
# Test SSH connectivity to hosts
for host in 192.222.51.29 192.222.51.167; do
  ssh -o ConnectTimeout=5 ubuntu@$host 'echo OK' && echo "$host: OK"
done

# Check host configuration
cat config/remote_hosts.yaml | grep ssh_host

# Manual sync test
rsync -avz ubuntu@192.222.51.29:~/ringrift/ai-service/data/games/*.db /tmp/
```

### Training Not Triggering

```bash
# Check games pending
curl -s http://209.20.157.81:9090/metrics | grep games_pending

# Verify threshold in config
grep trigger_threshold config/unified_loop.yaml

# Check training state
ssh ubuntu@209.20.157.81 'cat ~/ringrift/ai-service/logs/unified_loop/unified_loop_state.json | jq .training_in_progress'
```

### Metrics Endpoint Not Responding

```bash
# Check if prometheus_client is installed
ssh ubuntu@209.20.157.81 'python3 -c "from prometheus_client import Counter"'

# Check if port is in use
ssh ubuntu@209.20.157.81 'netstat -tlnp | grep 9090'

# Restart with fresh port
# Edit config/unified_loop.yaml: metrics_port: 9091
```

### Host Showing as Failed

```bash
# Check consecutive failures in state
ssh ubuntu@209.20.157.81 'cat ~/ringrift/ai-service/logs/unified_loop/unified_loop_state.json | \
  jq ".hosts | to_entries[] | select(.value.consecutive_failures > 0)"'

# Reset failure count by restarting loop
ssh ubuntu@209.20.157.81 'cd ~/ringrift/ai-service && python3 scripts/unified_ai_loop.py --stop'
ssh ubuntu@209.20.157.81 'cd ~/ringrift/ai-service && python3 scripts/unified_ai_loop.py --start'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    lambda_h100 (Primary)                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Unified AI Loop Daemon                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │  Data    │ │ Shadow   │ │ Training │ │  Model   │   │   │
│  │  │ Collector│ │Tournament│ │ Scheduler│ │ Promoter │   │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │   │
│  │       │            │            │            │          │   │
│  │       └────────────┴─────┬──────┴────────────┘          │   │
│  │                          │                               │   │
│  │                    ┌─────┴─────┐                        │   │
│  │                    │ Event Bus │                        │   │
│  │                    └───────────┘                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                    ┌─────────┴─────────┐                       │
│                    │ Prometheus :9090  │                       │
│                    └───────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
    ┌──────────┐         ┌──────────┐         ┌──────────┐
    │ gh200_a  │         │ gh200_b  │         │ gh200_c  │
    │ Selfplay │         │ Selfplay │         │ Selfplay │
    │ Workers  │         │ Workers  │         │ Workers  │
    └──────────┘         └──────────┘         └──────────┘
```

## Cluster Utilization Optimization

The unified loop includes intelligent resource optimization targeting 60-80% CPU/GPU utilization for optimal training throughput.

### Overview

The resource optimizer provides:

- **PID Controller** - Smooth workload adjustment using proportional-integral-derivative control
- **Rate Negotiation** - Cross-orchestrator coordination for selfplay rates
- **GPU Memory Tracking** - Automatic throttling when GPU memory is constrained
- **Predictive Scaling** - Proactive adjustments based on utilization trends
- **Config Weighting** - Balanced selfplay distribution across game configurations

### Configuration

Resource targets are configured in `config/unified_loop.yaml`:

```yaml
resource_targets:
  # CPU utilization targets (%)
  cpu_min: 60 # Below this, scale up jobs
  cpu_target: 70 # Ideal operating point
  cpu_max: 80 # Above this, throttle new jobs
  cpu_critical: 90

  # GPU utilization targets (%)
  gpu_min: 60
  gpu_target: 75 # Slightly higher than CPU
  gpu_max: 85
  gpu_critical: 95

  # PID controller tuning
  pid:
    kp: 0.3 # Proportional gain
    ki: 0.05 # Integral gain
    kd: 0.1 # Derivative gain
    gain_scheduling: true # Adjust gains based on error magnitude
    output_smoothing: 0.3 # Reduce sudden changes
```

### Monitoring Utilization

Key metrics exposed at `/metrics`:

| Metric                                     | Description                  |
| ------------------------------------------ | ---------------------------- |
| `ringrift_cluster_cpu_utilization_percent` | Cluster-wide CPU utilization |
| `ringrift_cluster_gpu_utilization_percent` | Cluster-wide GPU utilization |
| `ringrift_cluster_gpu_memory_utilization`  | GPU VRAM utilization         |
| `ringrift_gpu_memory_status`               | 0=ok, 1=warning, 2=critical  |
| `ringrift_optimization_action`             | Current scaling action       |
| `ringrift_cpu_in_target_range`             | 1 if CPU in 60-80% range     |
| `ringrift_gpu_in_target_range`             | 1 if GPU in 60-80% range     |

### Grafana Dashboard

The unified AI loop dashboard includes a "Cluster Utilization" section with:

- CPU/GPU utilization gauges (green = 60-80%, yellow/red = outside range)
- Selfplay rate tracking
- Utilization over time chart
- Config weight distribution

### Prometheus Alerts

Utilization alerts in `monitoring/prometheus/rules/utilization_alerts.yml`:

| Alert                     | Condition                | Severity |
| ------------------------- | ------------------------ | -------- |
| `CPUUnderutilized`        | CPU < 60% for 30m        | Warning  |
| `CPUOverutilized`         | CPU > 80% for 30m        | Warning  |
| `GPUUnderutilized`        | GPU < 60% for 30m        | Warning  |
| `GPUOverutilized`         | GPU > 80% for 30m        | Warning  |
| `GPUCriticallyOverloaded` | GPU > 95% for 10m        | Critical |
| `HighMemoryUtilization`   | Memory > 85% for 30m     | Warning  |
| `ProlongedBackpressure`   | Backpressure active > 1h | Warning  |

### Predictive Scaling

The system predicts future utilization and proactively adjusts:

```python
from app.coordination.resource_optimizer import (
    get_prediction,
    get_proactive_adjustment,
    apply_proactive_adjustment,
)

# Get utilization prediction
prediction = get_prediction()
# {
#   "predicted_cpu": 75.2,
#   "predicted_gpu": 72.8,
#   "cpu_trend": "rising",
#   "confidence": 0.85,
# }

# Apply proactive adjustment (call periodically)
new_rate = apply_proactive_adjustment()
```

### GPU Memory Management

GPU memory is tracked to prevent OOM errors:

- **Warning threshold (80%)**: Reduce new GPU job rate by 30%
- **Critical threshold (90%)**: Aggressive throttling to 30% of requested rate

Status is available in the utilization status API:

```python
from app.coordination.resource_optimizer import get_utilization_status

status = get_utilization_status()
print(status["gpu_memory"]["status"])  # "ok", "warning", or "critical"
```

### API Reference

```python
from app.coordination.resource_optimizer import (
    # Rate negotiation
    negotiate_selfplay_rate,
    get_current_selfplay_rate,
    apply_feedback_adjustment,

    # Utilization status
    get_utilization_status,
    get_optimization_recommendation,

    # Config weights
    update_config_weights,
    get_config_weights,

    # Predictive scaling
    get_prediction,
    apply_proactive_adjustment,
)

# Negotiate a selfplay rate
approved_rate = negotiate_selfplay_rate(
    requested_rate=1000,
    reason="startup",
    requestor="unified_loop"
)

# Apply feedback adjustment (call every ~30s)
new_rate = apply_feedback_adjustment(requestor="unified_loop")

# Get current status
status = get_utilization_status()
# {
#   "cpu_util": 72.5,
#   "gpu_util": 68.3,
#   "gpu_memory_util": 45.2,
#   "status": "optimal",
#   "current_rate": 850,
#   "recommendation": "OPTIMAL: Utilization 70% within 60-80% target range"
# }
```

## Files Reference

| File                                                 | Purpose                        |
| ---------------------------------------------------- | ------------------------------ |
| `scripts/unified_ai_loop.py`                         | Main daemon coordinator        |
| `scripts/deploy_unified_loop.sh`                     | Cluster deployment script      |
| `scripts/regression_gate.py`                         | Pre-promotion regression tests |
| `config/unified_loop.yaml`                           | Main configuration             |
| `config/remote_hosts.yaml`                           | Host definitions               |
| `config/systemd/ringrift-ai-loop.service`            | Systemd service file           |
| `config/monitoring/grafana-dashboard.json`           | Grafana dashboard              |
| `config/monitoring/alerting-rules.yaml`              | Prometheus alert rules         |
| `app/coordination/resource_optimizer.py`             | Resource utilization optimizer |
| `app/coordination/resource_targets.py`               | Utilization target definitions |
| `monitoring/prometheus/rules/utilization_alerts.yml` | Utilization-specific alerts    |
| `deploy/grafana/unified-ai-loop-dashboard.json`      | Main Grafana dashboard         |
| `logs/unified_loop/daemon.log`                       | Daemon log file                |
| `logs/unified_loop/unified_loop_state.json`          | Persistent state               |
| `data/coordination/resource_state.db`                | Resource optimizer state DB    |

## Command Reference

```bash
# Start/stop daemon
python3 scripts/unified_ai_loop.py --start
python3 scripts/unified_ai_loop.py --stop
python3 scripts/unified_ai_loop.py --foreground -v  # Run in foreground

# Check status
python3 scripts/unified_ai_loop.py --status

# Deploy to cluster
./scripts/deploy_unified_loop.sh
./scripts/deploy_unified_loop.sh --install-systemd --restart

# Run regression tests
python3 scripts/regression_gate.py --model <path> --config <config> -v

# Check metrics
curl http://209.20.157.81:9090/metrics | grep ringrift
curl http://209.20.157.81:9090/health
```
