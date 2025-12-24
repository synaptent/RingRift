# Obsolete Tests Archive

Tests moved here reference scripts that no longer exist.

## Archived Dec 24, 2025

### test_nn_training_baseline.py

- **Reason**: Imports `scripts.run_nn_training_baseline` which doesn't exist
- **Original purpose**: Tests for baseline Square-8 2-player NN training

### test_nn_training_baseline_demo.py

- **Reason**: Same as above
- **Original purpose**: Demo/smoke tests for baseline training

### test_tier_pipeline_scripts.py

- **Reason**: Imports `scripts.run_tier_training_pipeline` which doesn't exist
- **Original purpose**: Tests for tiered training pipeline scripts

## Replacement

Training is now handled by:

- `python -m app.training.train` - Main training CLI
- `python scripts/run_training_loop.py` - One-command training loop
- `python -m app.training.train_gnn_policy` - GNN training
