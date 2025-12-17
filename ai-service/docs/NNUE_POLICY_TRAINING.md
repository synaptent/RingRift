# NNUE Policy Training Guide

This document describes the NNUE policy training system, including KL divergence loss, MCTS data integration, and advanced training features.

## Overview

The NNUE policy training system trains neural networks with both value and policy heads. The policy head learns to predict moves from positions, while the value head predicts game outcomes.

**Key Script:** `scripts/train_nnue_policy.py`

## Quick Start

```bash
# Basic training from database
python scripts/train_nnue_policy.py \
    --db data/games/selfplay_square8_2p.db \
    --epochs 50

# Training with MCTS data (KL loss)
python scripts/train_nnue_policy.py \
    --jsonl data/selfplay/mcts_games.jsonl \
    --auto-kl-loss \
    --epochs 50

# Combined database + JSONL training
python scripts/train_nnue_policy.py \
    --db data/games/*.db \
    --jsonl data/selfplay/mcts_*.jsonl \
    --auto-kl-loss \
    --epochs 100
```

## Data Sources

### SQLite Databases (`--db`)

Standard game databases with positions and moves played:

```bash
--db data/games/selfplay_square8_2p.db
--db "data/games/*.db"  # Glob patterns supported
```

### JSONL Files with MCTS Data (`--jsonl`)

JSONL files containing games with MCTS visit distributions enable KL divergence loss:

```bash
--jsonl data/selfplay/mcts_square8_2p/games.jsonl
--jsonl "data/selfplay/mcts_*/games.jsonl"
```

JSONL format includes per-move MCTS policy:

```json
{
  "moves": [
    {
      "move": "d4",
      "mcts_policy": {"d4": 0.45, "e4": 0.30, "c4": 0.15, ...}
    }
  ]
}
```

## KL Divergence Loss

### What is KL Loss?

KL (Kullback-Leibler) divergence loss trains the policy head to match MCTS visit distributions rather than just predicting the single move played. This provides richer training signal from search results.

**Benefits:**

- Learns from full search distribution, not just final move
- Better policy calibration for uncertain positions
- Improved move ranking, not just top-1 accuracy

### Auto-KL Detection (`--auto-kl-loss`)

Automatically enables KL loss when sufficient MCTS data is available:

```bash
python scripts/train_nnue_policy.py \
    --jsonl data/selfplay/mcts_games.jsonl \
    --auto-kl-loss \
    --kl-min-coverage 0.3 \   # Min 30% of samples need MCTS data
    --kl-min-samples 50       # Min 50 samples with MCTS data
```

**Default thresholds:**

- `--kl-min-coverage 0.5` (50% coverage required)
- `--kl-min-samples 100` (100 samples minimum)

For early training with limited MCTS data, lower these:

```bash
--kl-min-coverage 0.3 --kl-min-samples 50
```

### Explicit KL Loss (`--use-kl-loss`)

Force KL loss regardless of coverage:

```bash
python scripts/train_nnue_policy.py \
    --jsonl data/selfplay/mcts_games.jsonl \
    --use-kl-loss
```

## Temperature Annealing

Controls label smoothness during training. Higher temperature = softer targets.

```bash
--temperature-start 2.0    # Start with soft targets
--temperature-end 0.5      # End with sharper targets
--temperature-schedule cosine  # linear, cosine, or exponential
```

**Schedules:**

- `linear`: Linear interpolation
- `cosine`: Smooth cosine decay (recommended)
- `exponential`: Rapid early decay

## Loss Weighting

Control the relative importance of policy and value losses:

```bash
--policy-weight 1.0   # Weight for policy loss (default: 1.0)
--value-weight 1.0    # Weight for value loss (default: 1.0)
```

To emphasize policy learning over value:

```bash
--policy-weight 2.0 --value-weight 0.5
```

## Advanced Training Features

### Gradient Accumulation

Simulate larger batches on limited GPU memory:

```bash
--gradient-accumulation-steps 4  # Accumulate 4 batches before optimizer step
```

Effective batch size = `batch_size * gradient_accumulation_steps`

### Progressive Batch Sizing

Start with small batches, grow larger:

```bash
--progressive-batch \
--min-batch-size 64 \
--max-batch-size 512
```

### Stochastic Weight Averaging (SWA)

Better generalization via weight averaging:

```bash
--use-swa \
--swa-start-epoch 30 \
--swa-lr 0.0001
```

### Exponential Moving Average (EMA)

Smooth model updates:

```bash
--use-ema \
--ema-decay 0.999
```

### Mixed Precision Training

Faster training with FP16:

```bash
--use-amp
```

### Focal Loss

Focus on hard samples:

```bash
--focal-gamma 2.0
```

### Label Smoothing Warmup

Gradually increase smoothing:

```bash
--label-smoothing 0.1 \
--label-smoothing-warmup 10  # Warmup over 10 epochs
```

## Curriculum Training

Train on move subsets by game phase:

```bash
# Early game only (moves 1-20)
--min-move-number 1 --max-move-number 20

# Late game only (moves 40+)
--min-move-number 40
```

## Policy Distillation

Train on winning player moves with boosted weight:

```bash
--distill-from-winners \
--winner-weight-boost 2.0 \
--min-winner-margin 0.1
```

## Hex Board Augmentation

D6 symmetry augmentation for hexagonal boards:

```bash
--hex-augment \
--hex-augment-count 6  # 1-12 augmentations
```

## Fine-tuning

### From Pretrained Model

```bash
python scripts/train_nnue_policy.py \
    --db data/games/new_data.db \
    --pretrained models/nnue/nnue_policy_square8_2p.pt \
    --epochs 20
```

### Freeze Value Head

Train only policy head:

```bash
--pretrained models/nnue/nnue_square8_2p.pt \
--freeze-value
```

## Multi-GPU Training

Distributed Data Parallel:

```bash
# Single machine, multiple GPUs
torchrun --nproc_per_node=4 scripts/train_nnue_policy.py \
    --db data/games/*.db \
    --use-ddp
```

### Manual Rank Assignment

For custom distributed setups:

```bash
# Set explicit rank for DDP
python scripts/train_nnue_policy.py \
    --db data/games/*.db \
    --use-ddp \
    --ddp-rank 0  # Rank of this process (0 = master)
```

## Output

### Model Checkpoint

Saved to `models/nnue/nnue_policy_{board_type}_{num_players}p.pt`

### Learning Curve Plots

Save training/validation curve plots:

```bash
--save-curves  # Saves loss and accuracy plots to run directory
```

Generates `{run_dir}/learning_curves.png` with:

- Training loss over epochs
- Validation loss over epochs
- Policy accuracy over epochs

### Training Report

JSON report at `{run_dir}/nnue_policy_training_report.json`:

```json
{
  "board_type": "square8",
  "num_players": 2,
  "dataset_size": 10000,
  "best_val_loss": 1.234,
  "final_val_policy_accuracy": 0.75,
  "history": {
    "train_loss": [...],
    "val_policy_accuracy": [...]
  }
}
```

## Environment Variables

| Variable                          | Description                   | Default |
| --------------------------------- | ----------------------------- | ------- |
| `RINGRIFT_POLICY_AUTO_KL_LOSS`    | Enable auto-KL detection      | `1`     |
| `RINGRIFT_POLICY_KL_MIN_COVERAGE` | Min MCTS coverage for auto-KL | `0.3`   |
| `RINGRIFT_POLICY_KL_MIN_SAMPLES`  | Min samples for auto-KL       | `50`    |

## Integration with Orchestrators

### Multi-Config Training Loop

The `scripts/multi_config_training_loop.py` automatically passes JSONL data to policy training:

```python
# In multi_config_training_loop.py
ENABLE_POLICY_TRAINING = True
POLICY_AUTO_KL_LOSS = os.environ.get("RINGRIFT_POLICY_AUTO_KL_LOSS", "1") == "1"
```

### Unified AI Loop

The unified loop enables policy training with:

```yaml
# config/unified_loop.yaml
policy_training:
  enabled: true
  auto_kl_loss: true
  kl_min_coverage: 0.3
```

## MCTS Data Generation

Generate MCTS data for KL loss training:

```bash
# Run MCTS selfplay
python scripts/run_mcts_balanced_selfplay.py \
    --board-type square8 \
    --num-players 2 \
    --mcts-sims 800 \
    --output data/selfplay/mcts_square8_2p/games.jsonl

# Reanalyze existing games with MCTS
python scripts/reanalyze_mcts_policy.py \
    --input data/games/existing.jsonl \
    --output data/games/with_mcts.jsonl \
    --mcts-sims 400
```

## Troubleshooting

### KL Loss Not Enabled

Check coverage thresholds:

```bash
# Lower thresholds for limited data
--kl-min-coverage 0.2 --kl-min-samples 30
```

### OOM Errors

Reduce batch size or enable gradient checkpointing:

```bash
--batch-size 128
--use-amp  # Uses less memory
```

### Slow Training

Enable mixed precision and parallel data loading:

```bash
--use-amp
--num-workers 4
```

## See Also

- `docs/TRAINING_PIPELINE.md` - Full training pipeline
- `docs/MCTS_INTEGRATION.md` - MCTS system details
- `scripts/train_nnue_policy_curriculum.py` - Curriculum-based training
