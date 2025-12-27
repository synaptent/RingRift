# Training Enhancements

Modular training enhancement components for RingRift AI.

This package was extracted from `training_enhancements.py` during the December 2025 modularization effort. Each module provides a focused capability that can be composed into training pipelines.

## Modules

### Core Configuration

| Module               | Class            | Description                                                |
| -------------------- | ---------------- | ---------------------------------------------------------- |
| `training_config.py` | `TrainingConfig` | Unified training hyperparameters with CLI override support |

### Gradient & Optimization

| Module                        | Class                     | Description                                            |
| ----------------------------- | ------------------------- | ------------------------------------------------------ |
| `gradient_management.py`      | `GradientAccumulator`     | Gradient accumulation for effective larger batch sizes |
| `gradient_management.py`      | `AdaptiveGradientClipper` | Per-parameter gradient clipping with moving average    |
| `learning_rate_scheduling.py` | `AdaptiveLRScheduler`     | LR scheduling with plateau detection and warmup        |
| `learning_rate_scheduling.py` | `WarmRestartsScheduler`   | Cosine annealing with warm restarts                    |

### Data Quality & Sampling

| Module                    | Class                    | Description                                      |
| ------------------------- | ------------------------ | ------------------------------------------------ |
| `data_quality_scoring.py` | `DataQualityScorer`      | Score training samples by quality metrics        |
| `data_quality_scoring.py` | `QualityWeightedSampler` | Importance-weighted sampling based on quality    |
| `hard_example_mining.py`  | `HardExampleMiner`       | Track and prioritize high-loss samples           |
| `per_sample_loss.py`      | `PerSampleLossTracker`   | Per-sample loss tracking for curriculum learning |

### Model Management

| Module                    | Class                | Description                                         |
| ------------------------- | -------------------- | --------------------------------------------------- |
| `checkpoint_averaging.py` | `CheckpointAverager` | EMA checkpoint averaging for stable models          |
| `model_ensemble.py`       | `ModelEnsemble`      | Ensemble multiple models for robust predictions     |
| `ewc_regularization.py`   | `EWCRegularizer`     | Elastic Weight Consolidation for continual learning |

### Evaluation & Feedback

| Module                   | Class                       | Description                                  |
| ------------------------ | --------------------------- | -------------------------------------------- |
| `evaluation_feedback.py` | `EvaluationFeedbackHandler` | Adjust training based on evaluation results  |
| `calibration.py`         | `CalibrationAutomation`     | Automated calibration for value head outputs |

### Infrastructure

| Module               | Class                        | Description                                     |
| -------------------- | ---------------------------- | ----------------------------------------------- |
| `seed_management.py` | `SeedManager`                | Deterministic seed management across frameworks |
| `training_facade.py` | `TrainingEnhancementsFacade` | Unified interface combining all enhancements    |

## Usage

### Direct Import

```python
from app.training.enhancements import (
    TrainingConfig,
    GradientAccumulator,
    HardExampleMiner,
)

config = TrainingConfig()
accumulator = GradientAccumulator(accumulation_steps=4)
miner = HardExampleMiner(buffer_size=1000)
```

### Via Facade (Recommended)

```python
from app.training.enhancements import TrainingEnhancementsFacade, FacadeConfig

facade = TrainingEnhancementsFacade(
    model=my_model,
    optimizer=my_optimizer,
    config=FacadeConfig(
        enable_quality_scoring=True,
        enable_hard_mining=True,
        gradient_accumulation_steps=4,
    )
)

# Training loop
for batch in dataloader:
    stats = facade.train_step(batch)
    if stats.should_stop:
        break
```

## Integration with Pipeline

These enhancements integrate with the broader training infrastructure:

- **Events**: `EvaluationFeedbackHandler` subscribes to `EVALUATION_COMPLETED` events
- **Daemons**: Quality metrics feed into `FeedbackLoopController`
- **Config**: All components respect `TrainingConfig` settings

## See Also

- `app/training/train.py` - Main training entry point
- `app/training/selfplay_runner.py` - Selfplay data generation
- `app/coordination/feedback_loop_controller.py` - Training feedback orchestration
