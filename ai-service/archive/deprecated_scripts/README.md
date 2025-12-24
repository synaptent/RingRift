# Deprecated Scripts Archive

Scripts moved here have been superseded by consolidated implementations.

## Archived Dec 24, 2025

### cluster_monitor_daemon.py

**Reason**: Functionality consolidated into `app/monitoring/unified_cluster_monitor.py`.
The unified monitor provides:

- Continuous monitoring with `start_monitoring(interval=60)`
- Event emission for status changes
- Health alerts with cooldown
- JSON/text formatting

Use: `python -m app.monitoring.unified_cluster_monitor --watch`

### robust_cluster_monitor.py

**Reason**: Functionality consolidated into `app/monitoring/unified_cluster_monitor.py`.
SSH-based node checking is now part of the unified monitor with HTTP fallback.

Use: `python -m app.monitoring.unified_cluster_monitor`

## Archived Dec 24, 2024

### export_replay_dataset_parallel.py

**Reason**: Parallel export is now the default in `export_replay_dataset.py`.
Use `--single-threaded` flag if sequential processing is needed.

### export_filtered_training.py

**Reason**: Quality filtering is available in `export_replay_dataset.py` via:

- `--require-completed`: Only completed games
- `--min-moves N`: Minimum move count
- `--max-moves N`: Maximum move count

## Active Export Scripts (Use These)

| Script                         | Purpose                     | Output Format |
| ------------------------------ | --------------------------- | ------------- |
| `export_replay_dataset.py`     | Main training data export   | NPZ           |
| `export_canonical_to_jsonl.py` | GMO training format         | JSONL         |
| `export_gumbel_kl_dataset.py`  | Soft policy KL distillation | NPZ           |
| `jsonl_to_npz.py`              | Convert JSONL to NPZ        | NPZ           |

## Usage Recommendations

1. **Standard NN Training**: Use `export_replay_dataset.py --use-discovery`
2. **GMO Training**: Use `export_canonical_to_jsonl.py`
3. **KL Distillation**: Use `export_gumbel_kl_dataset.py` on Gumbel MCTS JSONL
