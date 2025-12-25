# Scripts Index

**Last Updated:** December 24, 2025
**Total Scripts:** 386 (after archiving 74 debug/trace scripts)

## Quick Reference

| Category      | Count | Primary Use                           |
| ------------- | ----- | ------------------------------------- |
| `run_*`       | 63    | Execute pipelines, training, selfplay |
| `validate_*`  | 18    | Data and model validation             |
| `benchmark_*` | 14    | Performance benchmarking              |
| `analyze_*`   | 10    | Data analysis and statistics          |
| `check_*`     | 7     | Health and status checks              |
| `train_*`     | 6     | Training utilities                    |
| `export_*`    | 4     | Data export tools                     |
| `sync_*`      | 4     | Data synchronization                  |

## Core Workflows

### Training Pipeline

```bash
# Main training entry point
python -m app.training.train --board-type hex8 --num-players 2

# Automated training loop
python scripts/run_training_loop.py --board-type hex8 --num-players 2

# Export training data
python scripts/export_replay_dataset.py --use-discovery --output data/training/hex8_2p.npz
```

### Selfplay

```bash
# Unified selfplay entry point
python scripts/selfplay.py --board hex8 --num-players 2 --engine gumbel

# GPU selfplay
python scripts/run_gpu_selfplay.py --board-type hex8 --num-players 2
```

### Evaluation

```bash
# Gauntlet evaluation
python scripts/run_gauntlet.py --model models/hex8_2p.pth --board-type hex8

# Tournament
python scripts/run_tournament.py --board-type hex8 --num-players 2
```

### Cluster Operations

```bash
# Monitor cluster status
python -m app.distributed.cluster_monitor

# Sync data across cluster
python scripts/unified_data_sync.py --to-cluster
```

## Script Categories

### Runners (`run_*`) - 63 scripts

Primary execution scripts for pipelines and workflows.

**Key scripts:**

- `run_training_loop.py` - Automated training pipeline
- `run_gauntlet.py` - Model evaluation gauntlet
- `run_gpu_selfplay.py` - GPU-accelerated selfplay
- `run_tournament.py` - Tournament execution
- `run_distributed_training.py` - Multi-node training
- `run_canonical_selfplay_parity_gate.py` - Parity validation

### Validation (`validate_*`) - 18 scripts

Data integrity and model validation tools.

**Key scripts:**

- `validate_databases.py` - Database schema validation
- `validate_model_weights.py` - Model weight verification
- `validate_parity.py` - TS/Python parity validation

### Benchmarking (`benchmark_*`) - 14 scripts

Performance measurement and comparison.

**Key scripts:**

- `benchmark_gpu_mcts.py` - GPU MCTS performance
- `benchmark_ai_algorithms.py` - AI algorithm comparison
- `benchmark_engine.py` - Game engine performance

### Analysis (`analyze_*`) - 10 scripts

Data analysis and statistics.

**Key scripts:**

- `analyze_training_run.py` - Training metrics analysis
- `analyze_game_statistics.py` - Game data statistics
- `analyze_parity_failures.py` - Parity failure investigation

### Health Checks (`check_*`) - 7 scripts

System and data health verification.

**Key scripts:**

- `check_ts_python_replay_parity.py` - Parity gate check
- `check_cluster_health.py` - Cluster status check

### Training (`train_*`) - 6 scripts

Training-related utilities.

**Key scripts:**

- `train_nnue.py` - NNUE model training
- `train_heuristic_weights.py` - Heuristic weight optimization

### Export (`export_*`) - 4 scripts

Data export tools.

**Key scripts:**

- `export_replay_dataset.py` - Main export tool (supports --parallel, --use-discovery)

### Sync (`sync_*`) - 4 scripts

Data synchronization tools.

**Key scripts:**

- `sync_models.py` - Model synchronization
- `sync_databases.py` - Database synchronization

## Archived Scripts

### `archive/debug/` - 74 scripts

One-off debugging scripts archived on Dec 24, 2025.
See `archive/debug/README.md` for details.

### `archive/deprecated/` - Various

Deprecated scripts with documented replacements.
See individual README files in archive subdirectories.

## Adding New Scripts

When adding a new script:

1. Use appropriate prefix (`run_`, `validate_`, `benchmark_`, etc.)
2. Add docstring with usage example
3. Update this INDEX.md if it's a key workflow script
4. Consider if it should use existing infrastructure (e.g., `SelfplayRunner`, `ScriptRunner`)

## Script Infrastructure

Many scripts use shared infrastructure:

- `app/cli/runner.py` - `ScriptRunner` for argument parsing, logging, cleanup
- `app/cli/output.py` - Formatted console output
- `app/training/selfplay_runner.py` - Base class for selfplay scripts
- `scripts/lib/` - Shared utility functions
