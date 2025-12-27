# AI Service Scripts

This directory contains scripts for the RingRift AI training and improvement infrastructure.

## Canonical Entry Points

### Primary Orchestrator

**`master_loop.py`** - The canonical automation entry point. This is the main control plane for selfplay → sync → training → evaluation → promotion.

```bash
# Start the master loop (foreground)
python scripts/master_loop.py

# Watch status without starting the loop
python scripts/master_loop.py --watch

# Check status
python scripts/master_loop.py --status

# Legacy unified loop (explicit opt-in)
RINGRIFT_UNIFIED_LOOP_LEGACY=1 python scripts/unified_ai_loop.py --start
```

Features:

- Selfplay allocation across the cluster
- Training triggers with data quality gates
- Evaluation + promotion orchestration
- Unified data sync + model distribution hooks
- Feedback loop integration via the event router
- Profile-based daemon startup (minimal/standard/full)

## Key Categories

For the full script inventory, see `scripts/INDEX.md`.

### Cluster Management

- `cluster_health_check.py` - Cluster health snapshot
- `cluster_watchdog.py` - Host process watchdog
- `cluster_worker.py` - Worker node implementation
- `cluster_master_deploy.py` - Deploy orchestration helpers

### Training

- `run_training_loop.py` - Automated training loop
- `run_self_play_soak.py` - Self-play data generation
- `generate_canonical_selfplay.py` - Canonical self-play generator + gates
- `run_canonical_selfplay_parity_gate.py` - Canonical parity gate

### Evaluation

- `run_model_elo_tournament.py` - Model Elo tournaments
- `run_gauntlet.py` - Evaluation gauntlet
- `run_tournament.py` - Tournament runner

### Data Management

- `unified_data_sync.py` - **Unified data sync service** (replaces deprecated scripts below)
  - Run as daemon: `python scripts/unified_data_sync.py`
  - With watchdog: `python scripts/unified_data_sync.py --watchdog`
  - One-shot sync: `python scripts/unified_data_sync.py --once`
- `export_replay_dataset.py` - Export replay data to NPZ datasets
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

See `scripts/ARCHIVE_INDEX.md` and `scripts/DEPRECATED.md` for the curated list of archived and superseded scripts.

## Environment Variables

| Variable                         | Description                                 |
| -------------------------------- | ------------------------------------------- |
| `RINGRIFT_DISABLE_LOCAL_TASKS`   | Skip local training/eval (coordinator mode) |
| `RINGRIFT_TRACE_DEBUG`           | Enable detailed tracing                     |
| `RINGRIFT_SKIP_SHADOW_CONTRACTS` | Skip shadow contract validation             |
| `RINGRIFT_CONFIG_PATH`           | Override config path                        |
| `RINGRIFT_UNIFIED_LOOP_LEGACY`   | Enable legacy `unified_ai_loop.py`          |

## Configuration

`master_loop.py` reads configuration from `config/unified_loop.yaml`. Key settings:

- Data sync intervals
- Training thresholds
- Evaluation frequencies
- Cluster coordination options
