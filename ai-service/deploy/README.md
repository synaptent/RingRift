# RingRift AI Service - Deployment

> **Doc Status (2025-12-14): Active**

This directory contains deployment scripts and configurations for the RingRift AI training infrastructure.

## Directory Structure

```
deploy/
├── systemd/                    # Systemd service definitions (Linux)
│   ├── unified-ai-loop.service       # Main unified improvement loop
│   ├── model-promoter.service        # Elo-based model promotion
│   ├── shadow-tournament.service     # Lightweight evaluation tournaments
│   ├── streaming-data-collector.service  # Incremental data sync
│   ├── ringrift-p2p.service          # P2P cluster orchestrator
│   ├── ringrift-resilience.service   # Node resilience daemon
│   └── ringrift-improvement.service  # Legacy improvement loop
├── grafana/                    # Grafana dashboards
│   └── unified-ai-loop-dashboard.json
├── install-services.sh         # Install and enable systemd services
├── install-grafana-dashboard.sh # Set up Grafana monitoring
├── setup_node_resilience.sh    # Linux node setup script
├── setup_node_resilience_macos.sh  # macOS node setup script
├── deploy_cluster_resilience.py    # Cluster-wide deployment automation
└── RESILIENCE.md               # P2P resilience documentation
```

## Quick Start

### Option 1: Unified AI Loop (Recommended)

The unified AI loop combines all improvement components into a single daemon:

```bash
# Install the unified loop service
sudo ./install-services.sh unified-ai-loop

# Start it
sudo systemctl start unified-ai-loop
sudo systemctl enable unified-ai-loop

# Monitor logs
journalctl -u unified-ai-loop -f
```

### Option 2: Separate Services

Run individual services for more granular control:

```bash
# Install all services
sudo ./install-services.sh all

# Start individual services
sudo systemctl start streaming-data-collector
sudo systemctl start shadow-tournament
sudo systemctl start model-promoter
```

### Option 3: Node Resilience (P2P Cluster)

For distributed cluster deployment:

```bash
# Linux node setup
./setup_node_resilience.sh <node-id> <coordinator-url>

# macOS node setup
./setup_node_resilience_macos.sh <node-id> <coordinator-url>
```

See [RESILIENCE.md](RESILIENCE.md) for detailed P2P cluster setup.

## Services Overview

| Service                    | Port | Description                        |
| -------------------------- | ---- | ---------------------------------- |
| `unified-ai-loop`          | 9090 | All-in-one improvement coordinator |
| `streaming-data-collector` | -    | 60s incremental rsync from hosts   |
| `shadow-tournament`        | -    | 15min lightweight evaluation       |
| `model-promoter`           | -    | Auto-deploy on Elo threshold       |
| `ringrift-p2p`             | 8770 | P2P cluster control plane          |
| `ringrift-resilience`      | -    | Node health and fallback           |

## Monitoring

### Grafana Dashboard

```bash
# Install the dashboard
./install-grafana-dashboard.sh

# Access at: http://localhost:3000/d/unified-ai-loop
```

### Prometheus Metrics

The unified AI loop exports metrics on port 9090:

- `ringrift_games_synced_total` - Games synced per host
- `ringrift_training_runs_total` - Training run counter
- `ringrift_elo_rating` - Current Elo by model
- `ringrift_promotion_total` - Model promotions

### Health Endpoints

```bash
# Check unified loop health
curl http://localhost:9090/health

# Check P2P orchestrator
curl http://localhost:8770/health
```

## System Requirements

### Memory Requirements

All compute-intensive scripts require **64GB RAM minimum**:

| Script                       | Min RAM | Notes                         |
| ---------------------------- | ------- | ----------------------------- |
| `unified_ai_loop.py`         | 64GB    | Runs evaluations and training |
| `cluster_orchestrator.py`    | 64GB    | Filters hosts by memory       |
| `run_diverse_tournaments.py` | 64GB    | Filters hosts by memory       |
| `p2p_orchestrator.py`        | 64GB    | Filters nodes by memory       |

> **Note**: `continuous_improvement_daemon.py` is deprecated. Use `unified_ai_loop.py` instead.
> See [ORCHESTRATOR_SELECTION.md](../docs/infrastructure/ORCHESTRATOR_SELECTION.md) for guidance.

Scripts will exit with an error message if system memory is below the threshold.

### Environment Variables

| Variable                       | Default                 | Description                                |
| ------------------------------ | ----------------------- | ------------------------------------------ |
| `RINGRIFT_DISABLE_LOCAL_TASKS` | `false`                 | Skip local selfplay/training/tournaments   |
| `RINGRIFT_SYNC_STAGING`        | `false`                 | Sync promoted models to staging deployment |
| `USE_P2P_ORCHESTRATOR`         | `false`                 | Enable P2P cluster coordination            |
| `P2P_ORCHESTRATOR_URL`         | `http://localhost:8770` | P2P orchestrator endpoint                  |

**Low-Memory Dev Machines:**

To run coordination services on a machine with <64GB RAM:

```bash
# Add to ~/.zshrc or ~/.bashrc
export RINGRIFT_DISABLE_LOCAL_TASKS=true
```

This allows the daemon to run for data aggregation and monitoring without spawning compute tasks.

## Configuration

Services read configuration from:

- `/etc/ringrift/node.conf` - Node-specific settings
- `config/unified_loop.yaml` - Unified loop configuration
- `config/distributed_hosts.yaml` - Remote host definitions

## Model Rollback System

The rollback manager provides automated and manual rollback capabilities for production models.

**Module:** `app/training/rollback_manager.py`

### Usage

```python
from app.training.rollback_manager import RollbackManager, RollbackThresholds
from app.training.model_registry import ModelRegistry

registry = ModelRegistry()
manager = RollbackManager(registry)

# Set baseline after promoting a model
manager.set_baseline("square8_2p", {"elo": 1850, "win_rate": 0.62})

# Check if rollback is needed
should_rollback, reason = manager.should_rollback("square8_2p")
if should_rollback:
    result = manager.rollback_model("square8_2p", reason=reason)

# Manual rollback to specific version
result = manager.rollback_model("square8_2p", to_version=3, reason="Manual override")

# Get rollback history
history = manager.get_rollback_history(model_id="square8_2p", limit=10)
```

### Configurable Thresholds

```python
thresholds = RollbackThresholds(
    elo_drop_threshold=50.0,        # Elo points drop to trigger rollback
    elo_drop_window_hours=24.0,     # Time window to measure drop
    win_rate_drop_threshold=0.10,   # 10% win rate drop
    error_rate_threshold=0.05,      # 5% error rate
    min_games_for_evaluation=50,    # Minimum games before evaluating
)
manager = RollbackManager(registry, thresholds=thresholds)
```

### Prometheus Metrics

The rollback manager emits metrics for monitoring:

- `ringrift_model_rollbacks_total{model_id, trigger}` - Rollback counter by model and trigger type

Trigger types: `manual`, `auto_elo`, `auto_error`

### Alert Rules

Generate Prometheus alerting rules:

```python
from app.training.rollback_manager import create_rollback_alert_rules

alert_yaml = create_rollback_alert_rules()
with open("/etc/prometheus/rules/rollback.yml", "w") as f:
    f.write(alert_yaml)
```

**Alerts generated:**

- `ModelRollbackTriggered` - Warning when any rollback occurs
- `MultipleRollbacksDetected` - Critical when >2 rollbacks in 24h
- `EloDegradation` - Warning when Elo drops >50 from baseline

### Rollback History

Rollback events are persisted to `data/rollback_history.json` and include:

- Model ID and version transition
- Reason and trigger type
- Metrics before and after
- Timestamp and success status

```bash
# View recent rollbacks
cat data/rollback_history.json | jq '.[0:5]'
```

## Related Documentation

- [Orchestrator Selection](../docs/infrastructure/ORCHESTRATOR_SELECTION.md) - **Which script to use** (start here)
- [Unified AI Loop](../docs/training/UNIFIED_AI_LOOP.md) - Detailed loop documentation
- [Pipeline Orchestrator](../docs/infrastructure/PIPELINE_ORCHESTRATOR.md) - CI/CD pipeline orchestration
- [Distributed Selfplay](../docs/training/DISTRIBUTED_SELFPLAY.md) - Remote host setup
- [Cloud Infrastructure](../docs/infrastructure/CLOUD_TRAINING_INFRASTRUCTURE_PLAN.md) - AWS/cloud deployment
- [Cluster Monitoring](../scripts/monitoring/README.md) - CloudWatch setup

---

_Last updated: 2025-12-17_
