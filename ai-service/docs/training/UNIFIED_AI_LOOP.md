# Unified AI Self-Improvement Loop

> **Doc Status (2025-12-14): Active**
> **Location:** `scripts/unified_ai_loop.py`

The Unified AI Loop is a single coordinator daemon that integrates all components of the AI improvement cycle, replacing the need for multiple separate daemons.

## Overview

The unified loop coordinates five major subsystems:

| Component                     | Interval          | Purpose                                  |
| ----------------------------- | ----------------- | ---------------------------------------- |
| **Streaming Data Collection** | 30s               | Incremental rsync from all remote hosts  |
| **Shadow Tournament Service** | 5min              | Lightweight evaluation (15 games/config) |
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

# Enable Prometheus metrics on custom port
python scripts/unified_ai_loop.py --metrics-port 9091

# Disable metrics entirely
python scripts/unified_ai_loop.py --no-metrics

# Dry run (simulate without changes)
python scripts/unified_ai_loop.py --dry-run --verbose
```

### CLI Arguments Reference

| Argument         | Description                | Default                    |
| ---------------- | -------------------------- | -------------------------- |
| `--start`        | Start daemon in background | -                          |
| `--stop`         | Stop running daemon        | -                          |
| `--status`       | Show daemon status         | -                          |
| `--foreground`   | Run in foreground          | False                      |
| `--verbose`      | Enable verbose logging     | False                      |
| `--config`       | Path to config YAML        | `config/unified_loop.yaml` |
| `--halt`         | Set emergency halt flag    | -                          |
| `--resume`       | Clear emergency halt flag  | -                          |
| `--metrics-port` | Prometheus metrics port    | 9090                       |
| `--no-metrics`   | Disable Prometheus metrics | False                      |
| `--dry-run`      | Simulate without changes   | False                      |

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
  poll_interval_seconds: 30 # Check every 30s (optimized for fast feedback)
  sync_method: 'incremental' # "incremental" (rsync append) or "full"
  deduplication: true # Deduplicate games by ID
  min_games_per_sync: 5 # Sync smaller batches more frequently
  use_external_sync: true # Use unified_data_sync.py with P2P fallback

# Automatic training triggers
training:
  trigger_threshold_games: 300 # Start training when this many new games (optimized)
  min_interval_seconds: 1200 # 20 min between training runs (optimized)
  max_concurrent_jobs: 1 # Only one training job at a time
  prefer_gpu_hosts: true # Schedule training on GPU hosts

# Continuous evaluation
evaluation:
  shadow_interval_seconds: 300 # 5 min between shadow evals (optimized: 3x faster parallel)
  shadow_games_per_config: 15 # Games per shadow tournament (increased for lower variance)
  full_tournament_interval_seconds: 3600 # 1 hour between full tournaments

# Automatic model promotion
promotion:
  auto_promote: true # Enable automatic promotion
  elo_threshold: 20 # Must beat current best by this many Elo
  min_games: 40 # Minimum games before promotion eligible (Wilson CI provides safety)
  significance_level: 0.05 # Statistical significance requirement
  cooldown_seconds: 900 # 15 min cooldown between promotions (optimized)

# Adaptive curriculum (Elo-weighted training)
curriculum:
  adaptive: true # Enable adaptive curriculum
  rebalance_interval_seconds: 3600
  max_weight_multiplier: 1.5 # Reduced to avoid over-rotation

# Board/player configurations
configurations:
  - board_type: 'square8'
    num_players: [2, 3, 4]
  - board_type: 'square19'
    num_players: [2, 3, 4]
  - board_type: 'hexagonal'
    num_players: [2, 3, 4]
```

> **Note:** The above values are optimized defaults. See `config/unified_loop.yaml` for the complete configuration with all advanced options.

## Related Services

The unified loop can also be run as separate services if needed:

| Service           | Script                                 | Systemd Unit                |
| ----------------- | -------------------------------------- | --------------------------- |
| Data Collector    | `scripts/unified_data_sync.py`         | `unified-data-sync.service` |
| Shadow Tournament | `scripts/shadow_tournament_service.py` | `shadow-tournament.service` |
| Model Promoter    | `scripts/model_promotion_manager.py`   | `model-promoter.service`    |

> **Note:** `scripts/streaming_data_collector.py` is deprecated. Use `scripts/unified_data_sync.py` instead.

## Prometheus Metrics

When the Prometheus client is installed, the loop exports metrics on port 9090:

- `ringrift_games_synced_total` - Total games synced per host
- `ringrift_sync_duration_seconds` - Sync duration histogram
- `ringrift_training_runs_total` - Training runs counter
- `ringrift_elo_rating` - Current Elo ratings by model
- `ringrift_promotion_total` - Model promotions counter

## Data Flow

1. **Collection**: Every 30s, rsync pulls new games from all configured hosts
2. **Validation**: Games are validated against canonical gates before ingestion
3. **Tournament**: Every 5min, shadow tournaments evaluate model strength (15 games/config)
4. **Training**: When 300+ new games accumulated, training is auto-triggered
5. **Promotion**: If new model beats current by 20+ Elo, it's deployed (15min cooldown)
6. **Curriculum**: Training weights are adjusted based on Elo performance

## Coordinator-Only Mode

For machines that should only orchestrate the cluster without performing local compute-intensive tasks (selfplay, training, tournaments), set the `RINGRIFT_DISABLE_LOCAL_TASKS` environment variable:

```bash
# Enable coordinator-only mode
export RINGRIFT_DISABLE_LOCAL_TASKS=true

# Start the unified loop
python scripts/unified_ai_loop.py --start
```

### What Coordinator-Only Mode Disables

| Component              | Behavior in Coordinator Mode                      |
| ---------------------- | ------------------------------------------------- |
| **Local Selfplay**     | Skipped - games generated on cluster nodes only   |
| **Local Training**     | Skipped - training delegated to GPU nodes         |
| **Local Tournaments**  | Skipped - tournaments run on remote hosts         |
| **Data Collection**    | Active - syncs games from cluster nodes           |
| **Model Distribution** | Active - pushes models to cluster nodes           |
| **Metrics Export**     | Active - Prometheus metrics still available       |
| **Tournament Service** | Active - orchestrates remote tournament execution |

### Setting Up Persistent Coordinator Mode

Add to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
# RingRift: Run this machine as coordinator only
export RINGRIFT_DISABLE_LOCAL_TASKS=true
```

### Startup Message

When coordinator-only mode is enabled, the unified loop displays:

```
[UnifiedLoop] ════════════════════════════════════════════════════════════
[UnifiedLoop] COORDINATOR-ONLY MODE (RINGRIFT_DISABLE_LOCAL_TASKS=true)
[UnifiedLoop] Local selfplay, training, and tournaments will be delegated to cluster
[UnifiedLoop] ════════════════════════════════════════════════════════════
```

### Low-Memory Machines

On machines with less than 32GB RAM, the unified loop will suggest coordinator-only mode if not already set. This prevents OOM kills during memory-intensive operations.

## Related Documentation

- [Training Features](TRAINING_FEATURES.md) - Training configuration options
- [Training Triggers](TRAINING_TRIGGERS.md) - 3-signal trigger system
- [Training Internals](TRAINING_INTERNALS.md) - Internal training modules
- [Curriculum Feedback](CURRICULUM_FEEDBACK.md) - Adaptive curriculum weights
- [Training Optimizations](TRAINING_OPTIMIZATIONS.md) - Pipeline optimizations
- [Coordination System](../architecture/COORDINATION_SYSTEM.md) - Task coordination and resource management
- [Distributed Selfplay](DISTRIBUTED_SELFPLAY.md) - Remote host configuration
- [Pipeline Orchestrator](../infrastructure/PIPELINE_ORCHESTRATOR.md) - ⚠️ _Archived; replaced by unified loop_

---

_Last updated: 2025-12-17_
