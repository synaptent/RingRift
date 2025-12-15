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
> See [ORCHESTRATOR_SELECTION.md](../docs/ORCHESTRATOR_SELECTION.md) for guidance.

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

## Related Documentation

- [Orchestrator Selection](../docs/ORCHESTRATOR_SELECTION.md) - **Which script to use** (start here)
- [Unified AI Loop](../docs/UNIFIED_AI_LOOP.md) - Detailed loop documentation
- [Pipeline Orchestrator](../docs/PIPELINE_ORCHESTRATOR.md) - CI/CD pipeline orchestration
- [Distributed Selfplay](../docs/DISTRIBUTED_SELFPLAY.md) - Remote host setup
- [Cloud Infrastructure](../docs/CLOUD_TRAINING_INFRASTRUCTURE_PLAN.md) - AWS/cloud deployment
- [Cluster Monitoring](../scripts/monitoring/README.md) - CloudWatch setup

---

_Last updated: 2025-12-14_
