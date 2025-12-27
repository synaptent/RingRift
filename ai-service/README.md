# AI Service Scripts

This directory contains scripts for the RingRift AI training and improvement infrastructure.

## Canonical Entry Points

### Primary Orchestrator

**`unified_ai_loop.py`** - The canonical self-improvement orchestrator. This is the main entry point for the AI improvement loop.

```bash
# Start the unified loop
python scripts/unified_ai_loop.py --start

# Run in foreground with verbose output
python scripts/unified_ai_loop.py --foreground --verbose

# Check status
python scripts/unified_ai_loop.py --status
```

Features:

- Streaming data collection from distributed nodes (60s sync)
- Shadow tournament evaluation (15min lightweight)
- Training scheduler with data quality gates
- Model promotion with Elo thresholds
- Adaptive curriculum weighting
- Value calibration analysis
- Temperature scheduling for exploration control

## Key Categories

### Cluster Management

- `cluster_orchestrator.py` - Distributed cluster coordination
- `cluster_manager.py` - Cluster node management
- `cluster_worker.py` - Worker node implementation
- `cluster_control.py` - Cluster control commands
- `cluster_health_check.py` - Health monitoring

### Training

- `curriculum_training.py` - Generation-based curriculum training
- `run_self_play_soak.py` - Self-play data generation
- `run_hybrid_selfplay.py` - Hybrid self-play modes

### Evaluation

- `run_model_elo_tournament.py` - Model Elo tournaments
- `run_diverse_tournaments.py` - Multi-configuration tournaments
- `elo_promotion_gate.py` - Elo-based model promotion

### Data Management

- `unified_data_sync.py` - **Unified data sync service** (replaces deprecated scripts below)
  - Run as daemon: `python scripts/unified_data_sync.py`
  - With watchdog: `python scripts/unified_data_sync.py --watchdog`
  - One-shot sync: `python scripts/unified_data_sync.py --once`
- `streaming_data_collector.py` - _(DEPRECATED)_ Incremental game data sync - use `unified_data_sync.py`
- `collector_watchdog.py` - _(DEPRECATED)_ Collector health monitoring - use `unified_data_sync.py --watchdog`
- `sync_all_data.py` - _(DEPRECATED)_ Batch data sync - use `unified_data_sync.py --once`
- `build_canonical_training_pool_db.py` - Training data pooling
- `aggregate_jsonl_to_db.py` - JSONL to SQLite conversion

### Model Management

- `sync_models.py` - Model synchronization across cluster
- `prune_models.py` - Old model cleanup
- `model_promotion_manager.py` - Automated model promotion

### Analysis

- `analyze_game_statistics.py` - Game statistics analysis
- `check_ts_python_replay_parity.py` - TS/Python parity validation
- `track_elo_improvement.py` - Elo trend tracking

## Module Dependencies

### Canonical Service Interfaces

| Module                                | Purpose                | Usage                                        |
| ------------------------------------- | ---------------------- | -------------------------------------------- |
| `app.training.elo_service`            | Elo rating operations  | `get_elo_service()` singleton                |
| `app.training.curriculum`             | Curriculum training    | `CurriculumTrainer`, `CurriculumConfig`      |
| `app.training.value_calibration`      | Value head calibration | `ValueCalibrator`, `CalibrationTracker`      |
| `app.training.temperature_scheduling` | Exploration control    | `TemperatureScheduler`, `create_scheduler()` |

### Supporting Modules

| Module                                | Purpose                     |
| ------------------------------------- | --------------------------- |
| `app.tournament.elo`                  | Elo calculation utilities   |
| `app.training.elo_reconciliation`     | Distributed Elo consistency |
| `app.distributed.cluster_coordinator` | Cluster coordination        |
| `app.integration.pipeline_feedback`   | Training feedback loops     |

## Archived Scripts

The `archive/` subdirectory contains deprecated scripts that have been superseded:

| Script                                   | Superseded By                              |
| ---------------------------------------- | ------------------------------------------ |
| `master_self_improvement.py`             | `unified_ai_loop.py`                       |
| `unified_improvement_controller.py`      | `unified_ai_loop.py`                       |
| `integrated_self_improvement.py`         | `unified_ai_loop.py`                       |
| `export_replay_dataset.py`               | Direct DB queries                          |
| `validate_canonical_training_sources.py` | Data quality gates in `unified_ai_loop.py` |

## Environment Variables

| Variable                         | Description                                 |
| -------------------------------- | ------------------------------------------- |
| `RINGRIFT_DISABLE_LOCAL_TASKS`   | Skip local training/eval (coordinator mode) |
| `RINGRIFT_TRACE_DEBUG`           | Enable detailed tracing                     |
| `RINGRIFT_SKIP_SHADOW_CONTRACTS` | Skip shadow contract validation             |

## Configuration

The unified loop reads configuration from `config/unified_loop.yaml`. Key settings:

- Data sync intervals
- Training thresholds
- Evaluation frequencies
- Cluster coordination options
