# Training Pipeline Optimizations

This document covers the optimizations added to the training pipeline to improve speed, reliability, and adaptability.

## Overview

The training pipeline includes 7 key optimization systems:

| System              | Module                   | Purpose                       |
| ------------------- | ------------------------ | ----------------------------- |
| Export Cache        | `export_cache.py`        | Skip unchanged data exports   |
| Dynamic Export      | `dynamic_export.py`      | Auto-compute optimal settings |
| Curriculum Feedback | `curriculum_feedback.py` | Adaptive training weights     |
| Training Triggers   | `training_triggers.py`   | Simplified 3-signal system    |
| Distributed Locks   | `distributed_lock.py`    | Prevent concurrent conflicts  |
| Health Monitor      | `training_health.py`     | Track status and alerts       |
| Model Registry      | `training_registry.py`   | Full model traceability       |

## Export Cache

Skips re-exporting training data when source databases haven't changed.

### How It Works

```python
from app.training.export_cache import ExportCache

cache = ExportCache()

# Check if export is needed
if cache.needs_export(
    db_paths=["data/games/selfplay.db"],
    output_path="data/training/square8_2p.npz",
    board_type="square8",
    num_players=2,
):
    # Run export
    export_data(...)

    # Record in cache
    cache.record_export(
        db_paths=["data/games/selfplay.db"],
        output_path="data/training/square8_2p.npz",
        board_type="square8",
        num_players=2,
        samples_exported=500000,
    )
else:
    print("Export cache valid, skipping")
```

### Cache Invalidation

The cache is invalidated when:

- Source DB modification time changes
- Source DB game count increases
- Output file is deleted
- `force=True` is passed

### Performance

- **Typical speedup**: 10-30% faster training cycles
- **Cache location**: `data/training/.export_cache.json`

## Dynamic Export Settings

Automatically computes optimal export settings based on available data.

### Data Tiers

| Tier   | Game Count | max_games | sample_every | epochs | batch_size |
| ------ | ---------- | --------- | ------------ | ------ | ---------- |
| small  | <10K       | None      | 1            | 100    | 128        |
| medium | 10K-50K    | 50K       | 1            | 75     | 256        |
| large  | 50K-200K   | 100K      | 2            | 50     | 512        |
| xlarge | >200K      | 150K      | 3            | 30     | 1024       |

### Usage

```python
from app.training.dynamic_export import get_export_settings

settings = get_export_settings(
    db_paths=["data/games/selfplay.db"],
    board_type="square8",
    num_players=2,
)

print(f"Tier: {settings.data_tier}")
print(f"Max games: {settings.max_games}")
print(f"Sample every: {settings.sample_every}")
print(f"Epochs: {settings.epochs}")
print(f"Estimated samples: {settings.estimated_samples}")
```

### Board Adjustments

Different boards have different moves per game:

| Board     | Moves/Game | Sample Factor |
| --------- | ---------- | ------------- |
| square8   | 60         | 1.0x          |
| square19  | 200        | 1.5x          |
| hexagonal | 80         | 1.2x          |
| hex8      | 40         | 0.8x          |

## Curriculum Feedback

Tracks selfplay performance and adjusts training weights in real-time.

### Weight Calculation

Weights range from 0.5 (de-prioritize) to 2.0 (high priority):

```python
from app.training.curriculum_feedback import get_curriculum_feedback

cf = get_curriculum_feedback()

# Record game results
cf.record_game("square8_2p", winner=1, model_elo=1650)
cf.record_game("square8_2p", winner=-1, model_elo=1640)

# Get weights
weights = cf.get_curriculum_weights()
# {"square8_2p": 1.2, "hexagonal_2p": 0.8, ...}
```

### Weight Factors

| Factor               | Effect                  |
| -------------------- | ----------------------- |
| Low win rate (<55%)  | +0.4 weight             |
| No trained models    | +0.5 weight (bootstrap) |
| Elo regression       | +0.2 weight             |
| High win rate (>70%) | -0.3 weight             |
| Stale config (>6h)   | +0.1 weight             |

### Metrics Tracked

```python
metrics = cf.get_config_metrics("square8_2p")
print(f"Games (recent): {metrics.games_recent}")
print(f"Win rate: {metrics.win_rate:.1%}")
print(f"Elo trend: {metrics.elo_trend:+.1f}")
print(f"Model count: {metrics.model_count}")
```

## Training Triggers

Simplified 3-signal system replacing 8+ legacy signals.

### The 3 Signals

1. **Data Freshness**: New games since last training
2. **Model Staleness**: Time since last model update
3. **Performance Regression**: Win rate below threshold

### Usage

```python
from app.training.training_triggers import TrainingTriggers, TriggerConfig

triggers = TrainingTriggers(TriggerConfig(
    freshness_threshold=500,
    staleness_hours=6.0,
    min_win_rate=0.45,
))

# Update state
triggers.update_config_state(
    "square8_2p",
    games_count=600,
    win_rate=0.52,
    model_count=5,
)

# Check if training needed
decision = triggers.should_train("square8_2p")
print(f"Should train: {decision.should_train}")
print(f"Reason: {decision.reason}")
print(f"Priority: {decision.priority:.2f}")
```

### Signal Scores

```python
scores = decision.signal_scores
# {
#   "data_freshness": 0.8,     # 0-1 based on games
#   "model_staleness": 0.3,    # 0-1 based on time
#   "performance_regression": 0.0,  # 0-1 based on win rate
# }
```

## Distributed Locks

Prevents concurrent training on the same config across nodes.

### Redis Mode (Preferred)

```python
from app.coordination.distributed_lock import DistributedLock

lock = DistributedLock("training:square8_2p")

if lock.acquire(timeout=60):
    try:
        # Training code
        run_training()
    finally:
        lock.release()
```

### File Fallback

When Redis is unavailable, file-based locking is used:

- Lock files: `data/locks/training_square8_2p.lock`
- Automatic expiry via lock timeout
- Safe for single-node deployments

### Context Manager

```python
with DistributedLock("training:square8_2p") as lock:
    # Lock acquired automatically
    run_training()
# Lock released automatically
```

## Health Monitoring

Tracks training status and generates alerts.

### Recording Events

```python
from app.training.training_health import get_training_health_monitor

monitor = get_training_health_monitor()

# Record training events
monitor.record_training_start("square8_2p")
monitor.record_training_complete("square8_2p", success=True)
monitor.record_data_update("square8_2p", game_count=50000)
monitor.record_win_rate("square8_2p", win_rate=0.55)
```

### Health Status

```python
report = monitor.get_health_status()
print(f"Status: {report.status}")  # healthy/degraded/unhealthy
print(f"Summary: {report.summary}")

for alert in report.active_alerts:
    print(f"Alert: {alert.severity} - {alert.message}")
```

### Alert Types

| Alert                | Severity         | Condition             |
| -------------------- | ---------------- | --------------------- |
| Stalled training     | CRITICAL         | Running >4 hours      |
| Consecutive failures | WARNING/CRITICAL | 1-2 / 3+ failures     |
| Stale model          | WARNING          | No training >24 hours |
| Stale data           | WARNING          | No new data >12 hours |
| Low win rate         | WARNING          | Win rate <35%         |

### Prometheus Metrics

```python
metrics = monitor.get_prometheus_metrics()
# Returns Prometheus-formatted metrics text
```

Example output:

```
ringrift_training_is_running{config="square8_2p"} 0
ringrift_training_consecutive_failures{config="square8_2p"} 0
ringrift_training_model_count{config="square8_2p"} 15
ringrift_training_win_rate{config="square8_2p"} 0.55
ringrift_training_alerts_critical 0
ringrift_training_alerts_warning 1
```

## Model Registry

Tracks trained models with full lineage.

### Registration

```python
from app.training.training_registry import register_trained_model

model_id = register_trained_model(
    model_path="/path/to/model.pt",
    board_type="square8",
    num_players=2,
    training_config={
        "epochs": 50,
        "batch_size": 256,
        "learning_rate": 0.001,
    },
    metrics={
        "final_loss": 0.45,
        "best_val_loss": 0.43,
    },
    source_data_paths=["data/games/selfplay.db"],
    parent_model_id="model_v3_20241210",
)
print(f"Registered: {model_id}")
```

### Model Lookup

```python
from app.training.model_registry import ModelRegistry

registry = ModelRegistry(registry_dir="data/model_registry")

# Get model info
model = registry.get_model(model_id)
print(f"Created: {model.created_at}")
print(f"Status: {model.status}")
print(f"Training data hash: {model.data_hash}")
```

## Unified Pipeline

The `OptimizedTrainingPipeline` combines all optimizations:

```python
from app.training.optimized_pipeline import get_optimized_pipeline

pipeline = get_optimized_pipeline()

# Check status
status = pipeline.get_status()
print(f"Health: {status.health_status}")
print(f"Features: {status.available_features}")

# Run optimized training
result = pipeline.run_training(
    config_key="square8_2p",
    db_paths=["data/games/selfplay.db"],
)

print(f"Success: {result.success}")
print(f"Export time: {result.export_time:.1f}s")
print(f"Training time: {result.training_time:.1f}s")
print(f"Model ID: {result.model_id}")
```

## Integration Example

Complete training cycle with all optimizations:

```python
from app.training.optimized_pipeline import get_optimized_pipeline
from app.training.training_triggers import TrainingTriggers, TriggerConfig

# Initialize
pipeline = get_optimized_pipeline()
triggers = TrainingTriggers(TriggerConfig())

# Check if training needed
decision = triggers.should_train("square8_2p")

if decision.should_train:
    print(f"Training triggered: {decision.reason}")

    # Run with all optimizations
    result = pipeline.run_training(
        config_key="square8_2p",
        db_paths=["data/games/selfplay.db"],
    )

    if result.success:
        print(f"Training complete: {result.model_id}")
    else:
        print(f"Training failed: {result.message}")
```

## Related Documentation

- [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md) - Overall training workflow
- [FEEDBACK_ACCELERATOR.md](FEEDBACK_ACCELERATOR.md) - Momentum-based acceleration
- [TIER_PROMOTION_SYSTEM.md](TIER_PROMOTION_SYSTEM.md) - Model promotion
