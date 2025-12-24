# Training Documentation

Training pipeline and methodology documentation.

## Contents

| Document                                                                | Description                     |
| ----------------------------------------------------------------------- | ------------------------------- |
| [TRAINING_FEATURES](TRAINING_FEATURES.md)                               | Training feature specifications |
| [TRAINING_PIPELINE](TRAINING_PIPELINE.md)                               | Pipeline architecture           |
| [TRAINING_INTERNALS](TRAINING_INTERNALS.md)                             | Internal training modules       |
| [TRAINING_OPTIMIZATIONS](TRAINING_OPTIMIZATIONS.md)                     | Optimization techniques         |
| [TRAINING_TRIGGERS](TRAINING_TRIGGERS.md)                               | Training trigger system         |
| [CURRICULUM_FEEDBACK](CURRICULUM_FEEDBACK.md)                           | Curriculum learning             |
| [HYPERPARAMETER_OPTIMIZATION](HYPERPARAMETER_OPTIMIZATION.md)           | Hyperparameter tuning           |
| [FEEDBACK_ACCELERATOR](FEEDBACK_ACCELERATOR.md)                         | Feedback acceleration           |
| [DISTRIBUTED_SELFPLAY](DISTRIBUTED_SELFPLAY.md)                         | Distributed self-play           |
| [UNIFIED_AI_LOOP](UNIFIED_AI_LOOP.md)                                   | Main training loop              |
| [MODEL_REGISTRY](MODEL_REGISTRY.md)                                     | Model registry                  |
| [TIER_PROMOTION_SYSTEM](TIER_PROMOTION_SYSTEM.md)                       | Tier promotion                  |
| [PROMOTION_AND_ELO_RECONCILIATION](PROMOTION_AND_ELO_RECONCILIATION.md) | Elo reconciliation              |
| [EXAMPLE_TRAINING_RUN](EXAMPLE_TRAINING_RUN.md)                         | Example training run            |

## Quick Reference

### Transfer Learning (2p → 4p)

```bash
# Resize value head and train with transferred weights
python scripts/transfer_2p_to_4p.py \
  --source models/canonical_sq8_2p.pth \
  --output models/transfer_sq8_4p_init.pth \
  --board-type square8

python -m app.training.train \
  --board-type square8 --num-players 4 \
  --init-weights models/transfer_sq8_4p_init.pth \
  --data-path data/training/sq8_4p.npz
```

See [TRAINING_FEATURES.md#transfer-learning](TRAINING_FEATURES.md#transfer-learning) for details.
