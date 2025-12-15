# Unified AI Self-Improvement Loop

> **Doc Status (2025-12-14): Active**
> **Location:** `scripts/unified_ai_loop.py`

The Unified AI Loop is a single coordinator daemon that integrates all components of the AI improvement cycle, replacing the need for multiple separate daemons.

## Overview

The unified loop coordinates five major subsystems:

| Component                     | Interval          | Purpose                                  |
| ----------------------------- | ----------------- | ---------------------------------------- |
| **Streaming Data Collection** | 60s               | Incremental rsync from all remote hosts  |
| **Shadow Tournament Service** | 15min             | Lightweight evaluation (10 games/config) |
| **Training Scheduler**        | Threshold-based   | Auto-trigger when data thresholds met    |
| **Model Promoter**            | After tournaments | Auto-deploy on Elo threshold             |
| **Adaptive Curriculum**       | 1 hour            | Elo-weighted training focus              |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Unified AI Loop Daemon                      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Data       │  │   Shadow     │  │   Training   │       │
│  │   Collector  │──│   Tournament │──│   Scheduler  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│         │                │                  │                │
│         ▼                ▼                  ▼                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Model      │  │   Adaptive   │  │   Metrics    │       │
│  │   Promoter   │──│   Curriculum │──│   Exporter   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Command Line

```bash
# Start the unified loop (daemonized)
python scripts/unified_ai_loop.py --start

# Run in foreground with verbose output
python scripts/unified_ai_loop.py --foreground --verbose

# Check status
python scripts/unified_ai_loop.py --status

# Stop gracefully
python scripts/unified_ai_loop.py --stop

# Use custom config
python scripts/unified_ai_loop.py --config config/unified_loop.yaml

# Emergency halt (stops all loops at next health check, up to 5 min)
python scripts/unified_ai_loop.py --halt

# Resume after emergency halt
python scripts/unified_ai_loop.py --resume
```

### Emergency Halt

The unified loop supports an emergency halt mechanism for safely stopping all operations:

```bash
# Trigger emergency halt
python scripts/unified_ai_loop.py --halt

# Check if halt is active (shown in --status output)
python scripts/unified_ai_loop.py --status

# Clear halt flag to allow restart
python scripts/unified_ai_loop.py --resume
```

The halt flag is stored at `data/coordination/EMERGENCY_HALT`. When set:

- Running loops will stop at the next health check interval (every 5 minutes)
- New instances will refuse to start
- The flag persists across restarts until explicitly cleared

### Systemd Service

```bash
# Install the service
sudo cp deploy/systemd/unified-ai-loop.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable unified-ai-loop
sudo systemctl start unified-ai-loop

# Check status
sudo systemctl status unified-ai-loop
journalctl -u unified-ai-loop -f
```

## Configuration

Configuration is specified in `config/unified_loop.yaml`:

```yaml
# Data ingestion from remote hosts
data_ingestion:
  poll_interval_seconds: 60 # How often to check for new games
  sync_method: 'incremental' # "incremental" (rsync append) or "full"
  deduplication: true # Deduplicate games by ID
  min_games_per_sync: 10 # Only sync if at least this many new games

# Automatic training triggers
training:
  trigger_threshold_games: 1000 # Start training when this many new games
  min_interval_seconds: 1800 # At least 30 min between training runs
  max_concurrent_jobs: 1 # Only one training job at a time
  prefer_gpu_hosts: true # Schedule training on GPU hosts

# Continuous evaluation
evaluation:
  shadow_interval_seconds: 900 # 15 minutes between shadow evals
  shadow_games_per_config: 10 # Games per shadow tournament
  full_tournament_interval_seconds: 3600 # 1 hour between full tournaments

# Automatic model promotion
promotion:
  auto_promote: true # Enable automatic promotion
  elo_threshold: 20 # Must beat current best by this many Elo
  min_games: 50 # Minimum games before promotion eligible
  significance_level: 0.05 # Statistical significance requirement

# Adaptive curriculum (Elo-weighted training)
curriculum:
  adaptive: true # Enable adaptive curriculum
  rebalance_interval_seconds: 3600
  max_weight_multiplier: 2.0 # Max boost for underperforming configs

# Board/player configurations
configurations:
  - board_type: 'square8'
    num_players: [2, 3, 4]
  - board_type: 'square19'
    num_players: [2, 3, 4]
  - board_type: 'hexagonal'
    num_players: [2, 3, 4]
```

## Related Services

The unified loop can also be run as separate services if needed:

| Service           | Script                                 | Systemd Unit                       |
| ----------------- | -------------------------------------- | ---------------------------------- |
| Data Collector    | `scripts/streaming_data_collector.py`  | `streaming-data-collector.service` |
| Shadow Tournament | `scripts/shadow_tournament_service.py` | `shadow-tournament.service`        |
| Model Promoter    | `scripts/model_promotion_manager.py`   | `model-promoter.service`           |

## Prometheus Metrics

When the Prometheus client is installed, the loop exports metrics on port 9090:

- `ringrift_games_synced_total` - Total games synced per host
- `ringrift_sync_duration_seconds` - Sync duration histogram
- `ringrift_training_runs_total` - Training runs counter
- `ringrift_elo_rating` - Current Elo ratings by model
- `ringrift_promotion_total` - Model promotions counter

## Data Flow

1. **Collection**: Every 60s, rsync pulls new games from all configured hosts
2. **Validation**: Games are validated against canonical gates before ingestion
3. **Tournament**: Every 15min, shadow tournaments evaluate model strength
4. **Training**: When game threshold is met, training is auto-triggered
5. **Promotion**: If new model beats current by Elo threshold, it's deployed
6. **Curriculum**: Training weights are adjusted based on Elo performance

## Related Documentation

- [Pipeline Orchestrator](PIPELINE_ORCHESTRATOR.md) - Lower-level phase orchestration
- [Distributed Selfplay](DISTRIBUTED_SELFPLAY.md) - Remote host configuration
- [Training Data Registry](../TRAINING_DATA_REGISTRY.md) - Canonical data management
- [Self-Improvement Optimization](self_improvement_optimization_plan.md) - Performance tuning

---

_Last updated: 2025-12-14_
