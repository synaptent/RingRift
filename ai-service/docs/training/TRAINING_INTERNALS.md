# Training Internals

This document covers the internal modules that power the RingRift AI training pipeline.

## Overview

```
                    TRAINING PIPELINE

    +-------------------+     +-------------------+
    | Training Triggers |     | Value Calibration |
    +--------+----------+     +--------+----------+
             |                         |
             v                         v
    +-------------------+     +-------------------+
    | Curriculum        |     | Temperature       |
    | Trainer           |     | Scheduling        |
    +--------+----------+     +--------+----------+
             |                         |
             +------------+------------+
                          |
                          v
    +-----------------------------------------------------+
    |              UNIFIED AI LOOP                         |
    +-----------------------------------------------------+
    |  Orchestrates selfplay, training, evaluation        |
    |  Uses all modules for optimized training            |
    +-----------------------------------------------------+
```

## Training Triggers (`app/training/training_triggers.py`)

Simplified 3-signal system for deciding when to train.

### Core Signals

| Signal                     | Description                   | Default Threshold | Weight |
| -------------------------- | ----------------------------- | ----------------- | ------ |
| **Data Freshness**         | New games since last training | 300 games         | 1.0    |
| **Model Staleness**        | Hours since last training     | 6 hours           | 0.8    |
| **Performance Regression** | Win rate below threshold      | 0.45              | 1.5    |

### Usage

```python
from app.training.training_triggers import TrainingTriggers, TriggerConfig

# Configure triggers
config = TriggerConfig(
    freshness_threshold=300,  # Optimized for faster feedback
    staleness_hours=6.0,
    min_win_rate=0.45,
    min_interval_minutes=20,
)

triggers = TrainingTriggers(config)

# Check if training should run
decision = triggers.should_train("square8_2p", current_state)
if decision.should_train:
    print(f"Training triggered by: {decision.reason}")
    print(f"Priority: {decision.priority}")
```

### Bootstrap Mode

For new configurations with no models, a lower threshold (50 games) is used:

```python
config = TriggerConfig(
    bootstrap_threshold=50,  # Very low for configs with 0 models
)
```

### Configuration

| Parameter                 | Default | Description                         |
| ------------------------- | ------- | ----------------------------------- |
| `freshness_threshold`     | 300     | New games needed to trigger         |
| `staleness_hours`         | 6.0     | Hours before config is "stale"      |
| `min_win_rate`            | 0.45    | Below this triggers urgent training |
| `min_interval_minutes`    | 20      | Minimum time between runs           |
| `max_concurrent_training` | 3       | Max parallel training jobs          |
| `bootstrap_threshold`     | 50      | Threshold for new configs           |

## Curriculum Learning (`app/training/curriculum.py`)

Generation-based iterative training with self-play.

### Features

- Generation-based training loops
- Model evaluation and promotion
- Historical data mixing
- Position complexity estimation
- Stage-based sample filtering
- Adaptive difficulty adjustment

### Canonical Configurations

| Board     | Players | Status       | Notes            |
| --------- | ------- | ------------ | ---------------- |
| square8   | 2       | Primary      | Production focus |
| square19  | 2       | Experimental | Larger board     |
| hexagonal | 2       | Experimental | D6 symmetry      |

### Usage

```bash
# Launch curriculum training
cd ai-service
python -m app.training.curriculum \
  --board-type square8 \
  --generations 10 \
  --games-per-gen 1000 \
  --eval-games 100 \
  --output-dir curriculum_runs/square8_2p
```

### CurriculumConfig

```python
from app.training.curriculum import CurriculumTrainer, CurriculumConfig

config = CurriculumConfig(
    generations=10,
    games_per_generation=1000,
    eval_games=100,
    max_moves=200,
    historical_mix_ratio=0.3,  # 30% historical data
)

trainer = CurriculumTrainer(config)
trainer.run()
```

### Position Complexity Estimation

Curriculum learning uses complexity estimation to stage training samples. The complexity
estimation is handled internally by the `AdaptiveSchedule` class in temperature_scheduling:

```python
from app.training.temperature_scheduling import AdaptiveSchedule

# Create adaptive temperature schedule with complexity awareness
schedule = AdaptiveSchedule(
    base_temp=1.0,
    min_temp=0.1,
    max_temp=2.0,
)

# Get temperature adjusted for position complexity
# Internally calls _estimate_complexity(game_state) which returns 0.0-1.0
temperature = schedule.get_temperature(
    move_number=current_move,
    game_state=game_state,
)
```

Complexity factors considered:

- Piece count and distribution
- Board control percentages
- Potential winning lines
- Game phase (opening/mid/end)

## Value Calibration (`app/training/value_calibration.py`)

Ensures value predictions match actual game outcomes.

### Key Metrics

| Metric             | Description                | Good Value |
| ------------------ | -------------------------- | ---------- |
| **ECE**            | Expected Calibration Error | < 0.05     |
| **MCE**            | Maximum Calibration Error  | < 0.10     |
| **Overconfidence** | Systematic bias (-1 to 1)  | ~0.0       |
| **Brier Score**    | Mean squared error         | < 0.25     |

### Usage

```python
from app.training.value_calibration import ValueCalibrator, CalibrationReport

calibrator = ValueCalibrator(num_bins=10)

# Add predictions and outcomes
for pred, outcome in data:
    calibrator.add_sample(prediction=pred, outcome=outcome)

# Generate report
report: CalibrationReport = calibrator.compute_calibration()
print(f"ECE: {report.ece:.4f}")
print(f"MCE: {report.mce:.4f}")
print(f"Overconfidence: {report.overconfidence:.4f}")
print(f"Optimal Temperature: {report.optimal_temperature:.4f}")
```

### Calibration Bins

```python
# Access bin details
for bin in report.bins:
    print(f"Range [{bin.lower:.1f}, {bin.upper:.1f}]:")
    print(f"  Count: {bin.count}")
    print(f"  Mean Prediction: {bin.mean_prediction:.3f}")
    print(f"  Mean Outcome: {bin.mean_outcome:.3f}")
    print(f"  Calibration Error: {bin.calibration_error:.3f}")
```

### Temperature Scaling

When overconfidence is detected, apply temperature scaling:

```python
if report.optimal_temperature:
    # Scale logits by optimal temperature
    scaled_logits = logits / report.optimal_temperature
```

## Temperature Scheduling (`app/training/temperature_scheduling.py`)

Controls exploration vs exploitation during training.

### Schedulers

| Scheduler              | Pattern           | Use Case          |
| ---------------------- | ----------------- | ----------------- |
| `LinearScheduler`      | Linear decay      | Default           |
| `CosineScheduler`      | Cosine annealing  | Smooth decay      |
| `StepScheduler`        | Discrete steps    | Curriculum stages |
| `ExponentialScheduler` | Exponential decay | Fast convergence  |

### Usage

```python
from app.training.temperature_scheduling import (
    TemperatureScheduler,
    create_scheduler,
)

# Create scheduler
scheduler = create_scheduler(
    scheduler_type="cosine",
    start_temp=1.0,
    end_temp=0.1,
    total_steps=10000,
)

# Get temperature for current step
for step in range(total_steps):
    temperature = scheduler.get_temperature(step)
    # Apply to softmax: probs = softmax(logits / temperature)
```

### Scheduler Parameters

```python
# Linear scheduler
scheduler = LinearScheduler(
    start_temp=1.0,
    end_temp=0.1,
    warmup_steps=1000,  # Optional warmup
)

# Cosine scheduler with restarts
scheduler = CosineScheduler(
    start_temp=1.0,
    end_temp=0.1,
    period=5000,  # Restart period
    num_restarts=2,
)
```

## Integration with Unified Loop

All these modules integrate with `scripts/unified_ai_loop.py`:

```yaml
# config/unified_loop.yaml
training:
  triggers:
    freshness_threshold: 300 # Optimized for faster feedback
    staleness_hours: 6
    min_win_rate: 0.45

  calibration:
    enabled: true
    check_interval: 3600 # Check every hour
    ece_threshold: 0.10 # Retrain if ECE exceeds

  temperature:
    scheduler: cosine
    start: 1.0
    end: 0.1
```

## Environment Variables

| Variable                     | Default | Description           |
| ---------------------------- | ------- | --------------------- |
| `RINGRIFT_TRIGGER_FRESHNESS` | 300     | Games before training |
| `RINGRIFT_TRIGGER_STALENESS` | 6       | Hours before stale    |
| `RINGRIFT_TRIGGER_MIN_WIN`   | 0.45    | Min win rate          |
| `RINGRIFT_CALIBRATION_BINS`  | 10      | Calibration bins      |
| `RINGRIFT_TEMP_SCHEDULER`    | cosine  | Temperature scheduler |

## Monitoring

### Prometheus Metrics

```
# Training triggers
ringrift_trigger_freshness_score{config="square8_2p"} 0.75
ringrift_trigger_staleness_score{config="square8_2p"} 0.30
ringrift_trigger_regression_score{config="square8_2p"} 0.00

# Value calibration
ringrift_calibration_ece{config="square8_2p"} 0.042
ringrift_calibration_mce{config="square8_2p"} 0.085
ringrift_calibration_overconfidence{config="square8_2p"} 0.012

# Temperature
ringrift_current_temperature{config="square8_2p"} 0.45
```

## Related Documentation

- [TRAINING_FEATURES.md](TRAINING_FEATURES.md) - Training parameters and CLI flags
- [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md) - High-level pipeline overview
- [UNIFIED_AI_LOOP.md](UNIFIED_AI_LOOP.md) - Orchestrator documentation
- [scripts/README.md](../scripts/README.md) - Script reference
