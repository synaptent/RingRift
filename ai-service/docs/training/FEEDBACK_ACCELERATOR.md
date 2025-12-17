# Feedback Accelerator

The Feedback Accelerator optimizes the AI training loop by detecting improvement momentum and accelerating training when models are performing well. It creates a positive feedback cycle that maximizes learning efficiency.

## Core Concept

```
┌─────────────────────────────────────────────────────────────┐
│              POSITIVE FEEDBACK CYCLE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│     ┌──────────┐     ┌──────────┐     ┌──────────┐          │
│     │  Train   │────▶│  Eval    │────▶│  Elo     │          │
│     │  Model   │     │  Games   │     │  Update  │          │
│     └──────────┘     └──────────┘     └────┬─────┘          │
│          ▲                                  │                │
│          │           ┌──────────┐           │                │
│          └───────────│ Feedback │◀──────────┘                │
│                      │Accelerator│                           │
│                      └──────────┘                            │
│                            │                                 │
│                   Momentum Detection                         │
│                            │                                 │
│              ┌─────────────┼─────────────┐                   │
│              ▼             ▼             ▼                   │
│         ACCELERATING   IMPROVING     PLATEAU                 │
│         (hot path)     (normal+)    (analyze)                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Usage

```python
from app.training.feedback_accelerator import (
    FeedbackAccelerator,
    get_feedback_accelerator,
    should_trigger_training,
    get_training_intensity,
)

# Get singleton accelerator
accelerator = get_feedback_accelerator()

# Check if training should be triggered
if accelerator.should_trigger_training("square8_2p"):
    intensity = accelerator.get_training_intensity("square8_2p")
    print(f"Training with intensity: {intensity.multiplier}x")
    print(f"Momentum: {intensity.momentum_state}")

# Record Elo update after evaluation
accelerator.record_elo_update(
    config_key="square8_2p",
    new_elo=1650,
    games_played=100,
)
```

## Momentum States

The accelerator tracks 5 momentum states:

| State          | Elo Change     | Training Response                 |
| -------------- | -------------- | --------------------------------- |
| `ACCELERATING` | ≥+25 Elo       | Hot path (75 games threshold)     |
| `IMPROVING`    | +12 to +25 Elo | Accelerated (150 games threshold) |
| `STABLE`       | +5 to +12 Elo  | Normal (300 games threshold)      |
| `PLATEAU`      | -5 to +5 Elo   | Normal with analysis              |
| `REGRESSING`   | ≤-5 Elo        | Slow down, investigate            |

## Training Triggers

### Game Thresholds

The accelerator adjusts training triggers based on momentum:

```python
# Default thresholds (configurable via env vars)
MIN_GAMES_FOR_TRAINING = 300      # Normal training
ACCELERATED_MIN_GAMES = 150       # Improving models
HOT_PATH_MIN_GAMES = 75           # Accelerating models
```

### Trigger Check

```python
from app.training.feedback_accelerator import should_trigger_training

# Returns True if enough games and momentum warrants training
should_train = should_trigger_training(
    config_key="square8_2p",
    games_since_last_training=200,
)
```

## Training Intensity

The accelerator provides intensity multipliers for training parameters:

```python
@dataclass
class TrainingIntensity:
    multiplier: float          # 0.5x to 2.5x
    momentum_state: str        # Current momentum
    recommended_epochs: int    # Adjusted epochs
    recommended_lr_scale: float # Learning rate scale
```

### Intensity Effects

| Multiplier | Effect                        |
| ---------- | ----------------------------- |
| 2.5x       | Max epochs, aggressive LR     |
| 2.0x       | High epochs, boosted LR       |
| 1.5x       | Above-normal training         |
| 1.0x       | Standard training             |
| 0.5x       | Reduced training (regression) |

## Elo Momentum Tracking

The accelerator maintains a rolling window of Elo updates:

```python
# Record Elo update
accelerator.record_elo_update(
    config_key="square8_2p",
    new_elo=1650,
    games_played=100,
    timestamp=datetime.now(),
)

# Get momentum analysis
momentum = accelerator.get_momentum("square8_2p")
print(f"State: {momentum.state}")
print(f"Trend: {momentum.elo_trend:+.1f} per 100 games")
print(f"Lookback: {momentum.lookback_games} games")
```

## Hot Path Mode

When a model is ACCELERATING, the hot path enables:

1. **Reduced game threshold** (75 games vs 300)
2. **Priority scheduling** for training jobs
3. **Increased epoch count** (up to 2.5x)
4. **Faster promotion evaluation**

```python
# Check if config is on hot path
if accelerator.is_hot_path("square8_2p"):
    print("Model on hot path - maximum acceleration")
```

## Curriculum Weight Adjustment

The accelerator adjusts curriculum weights per config:

```python
# Get current curriculum weights
weights = accelerator.get_curriculum_weights()
# {"square8_2p": 1.5, "hexagonal_2p": 0.8, ...}

# Weights affect:
# - Selfplay game allocation
# - Training data sampling
# - Resource scheduling
```

## Persistence

State is persisted to SQLite for crash recovery:

```
data/feedback/accelerator_state.db
├── elo_history          # Elo update records
├── momentum_cache       # Computed momentum states
├── training_triggers    # Trigger timestamps
└── curriculum_weights   # Current weights
```

## Configuration

### Environment Variables

| Variable                      | Default | Description               |
| ----------------------------- | ------- | ------------------------- |
| `RINGRIFT_MIN_GAMES_TRAINING` | 300     | Normal training threshold |
| `RINGRIFT_ACCEL_MIN_GAMES`    | 150     | Accelerated threshold     |
| `RINGRIFT_HOT_MIN_GAMES`      | 75      | Hot path threshold        |
| `RINGRIFT_FEEDBACK_LOOKBACK`  | 5       | Elo updates for momentum  |

### Tuning Constants

```python
# In feedback_accelerator.py
ELO_MOMENTUM_LOOKBACK = 5       # Updates to consider
ELO_STRONG_IMPROVEMENT = 25.0   # Elo gain for ACCELERATING
ELO_MODERATE_IMPROVEMENT = 12.0 # Elo gain for IMPROVING
ELO_PLATEAU_THRESHOLD = 5.0     # Below this = PLATEAU
MAX_INTENSITY_MULTIPLIER = 2.5  # Max training boost
MIN_INTENSITY_MULTIPLIER = 0.5  # Min (for regression)
```

## Integration

### With Unified Loop

```python
# In unified_ai_loop.py
from app.training.feedback_accelerator import get_feedback_accelerator

accelerator = get_feedback_accelerator()

async def training_cycle():
    for config_key in configs:
        # Check momentum-aware trigger
        if accelerator.should_trigger_training(config_key):
            intensity = accelerator.get_training_intensity(config_key)
            await run_training(config_key, intensity)
```

### With Curriculum System

```python
# Weights flow to selfplay allocation
weights = accelerator.get_curriculum_weights()
for config, weight in weights.items():
    # Higher weight = more selfplay games
    selfplay_allocation[config] = base_games * weight
```

## Monitoring

### Prometheus Metrics

```
# Momentum metrics
ringrift_feedback_momentum_state{config="square8_2p"} 1  # 1=accelerating
ringrift_feedback_elo_trend{config="square8_2p"} 25.5
ringrift_feedback_training_intensity{config="square8_2p"} 2.0

# Trigger metrics
ringrift_feedback_hot_path_configs 2
ringrift_feedback_training_triggers_total{reason="hot_path"} 15
```

### Debug Commands

```bash
# View current momentum states
python -c "
from app.training.feedback_accelerator import get_feedback_accelerator
acc = get_feedback_accelerator()
for config in ['square8_2p', 'hexagonal_2p']:
    m = acc.get_momentum(config)
    print(f'{config}: {m.state}, trend={m.elo_trend:+.1f}')
"

# View Elo history
python -c "
from app.training.feedback_accelerator import get_feedback_accelerator
acc = get_feedback_accelerator()
history = acc.get_elo_history('square8_2p', limit=10)
for h in history:
    print(f'{h.timestamp}: Elo={h.elo}, games={h.games}')
"
```

## Troubleshooting

### Model Stuck on Hot Path

If a model remains on hot path but performance plateaus:

```python
# Force momentum recalculation
accelerator.recalculate_momentum("square8_2p")

# Or reset momentum state
accelerator.reset_momentum("square8_2p")
```

### Training Not Triggering

1. Check game count since last training
2. Verify momentum state isn't REGRESSING
3. Check for competing training locks

```bash
python -c "
from app.training.feedback_accelerator import get_feedback_accelerator
acc = get_feedback_accelerator()
config = 'square8_2p'
print(f'Should train: {acc.should_trigger_training(config)}')
print(f'Momentum: {acc.get_momentum(config)}')
print(f'Games since training: {acc.games_since_training(config)}')
"
```

### Corrupted State

```bash
# Reset accelerator state
rm data/feedback/accelerator_state.db
# State will be rebuilt from Elo database on next run
```

## Related Documentation

- [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md) - Training workflow
- [CURRICULUM_FEEDBACK.md](CURRICULUM_FEEDBACK.md) - Curriculum weights
- [TRAINING_TRIGGERS.md](TRAINING_TRIGGERS.md) - Trigger system
