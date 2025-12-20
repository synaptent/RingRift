# Training Triggers System

> **Last Updated**: 2025-12-17
> **Status**: Active
> **Location**: `app/training/training_triggers.py`

This document describes the simplified 3-signal training trigger system that determines when to initiate model training.

## Overview

The training triggers system consolidates training decision logic into 3 core signals, replacing a complex 8+ signal system with a cleaner, more predictable approach.

### The 3 Core Signals

| Signal                     | Description                             | Weight |
| -------------------------- | --------------------------------------- | ------ |
| **Data Freshness**         | New games available since last training | 1.0    |
| **Model Staleness**        | Time since last training for config     | 0.8    |
| **Performance Regression** | Elo/win rate below acceptable threshold | 1.5    |

## Quick Start

```python
from app.training.training_triggers import TrainingTriggers, TriggerConfig

# Create with default config
triggers = TrainingTriggers()

# Or with custom config
triggers = TrainingTriggers(TriggerConfig(
    freshness_threshold=300,  # Optimized for faster feedback loops
    staleness_hours=6.0,
    min_win_rate=0.45,
))

# Update state for a config
triggers.update_config_state(
    "square8_2p",
    games_count=600,
    win_rate=0.52,
    model_count=5,
)

# Check if training should run
decision = triggers.should_train("square8_2p")
if decision.should_train:
    print(f"Training triggered: {decision.reason}")
    print(f"Priority: {decision.priority:.2f}")
```

## Configuration

### TriggerConfig Parameters

| Parameter                 | Type  | Default | Description                             |
| ------------------------- | ----- | ------- | --------------------------------------- |
| `freshness_threshold`     | int   | 300     | New games needed to trigger             |
| `freshness_weight`        | float | 1.0     | Weight of freshness signal              |
| `staleness_hours`         | float | 6.0     | Hours before config is "stale"          |
| `staleness_weight`        | float | 0.8     | Weight of staleness signal              |
| `min_win_rate`            | float | 0.45    | Win rate threshold for regression       |
| `regression_weight`       | float | 1.5     | Weight of regression signal             |
| `min_interval_minutes`    | float | 20      | Minimum time between training runs      |
| `max_concurrent_training` | int   | 3       | Max parallel training jobs              |
| `bootstrap_threshold`     | int   | 50      | Low threshold for configs with 0 models |

### Example Configuration

```python
config = TriggerConfig(
    # Data freshness
    freshness_threshold=300,  # Optimized for faster feedback loops
    freshness_weight=1.0,

    # Model staleness
    staleness_hours=6.0,
    staleness_weight=0.8,

    # Performance regression
    min_win_rate=0.45,
    regression_weight=1.5,

    # Global constraints
    min_interval_minutes=20,
    max_concurrent_training=3,

    # Bootstrap for new configs
    bootstrap_threshold=50,
)
```

## Signal Details

### 1. Data Freshness

Measures new games available since last training.

**Score Calculation:**

```python
freshness_score = min(1.0, games_since_training / freshness_threshold)
```

**When it triggers:**

- Score reaches 1.0 when `games_since_training >= freshness_threshold`
- Lower threshold (50 games) for bootstrap configs with 0 models

### 2. Model Staleness

Measures time elapsed since last training.

**Score Calculation:**

```python
hours_since_training = (now - last_training_time) / 3600
staleness_score = min(1.0, hours_since_training / staleness_hours)
```

**When it triggers:**

- Score reaches 1.0 when `hours_since_training >= staleness_hours`
- Ensures configs don't go too long without updates

### 3. Performance Regression

Measures if model performance has degraded.

**Score Calculation:**

```python
if win_rate < min_win_rate:
    regression_score = (min_win_rate - win_rate) / min_win_rate
else:
    regression_score = 0.0
```

**When it triggers:**

- Non-zero when win rate drops below `min_win_rate` (default 45%)
- Highest weight (1.5x) for urgent recovery

## Decision Logic

### Priority Calculation

```python
priority = (
    freshness_score * freshness_weight +
    staleness_score * staleness_weight +
    regression_score * regression_weight
)
```

### Training Decision

Training is triggered when:

1. **Priority threshold exceeded**: `priority >= 1.0`
2. **Minimum interval passed**: `time_since_last >= min_interval_minutes`
3. **Bootstrap needed**: Config has 0 models and 50+ games
4. **Concurrent limit not exceeded**: Active training < `max_concurrent_training`

## API Reference

### TrainingTriggers Class

```python
class TrainingTriggers:
    """Simplified training trigger system with 3 core signals."""

    def __init__(self, config: Optional[TriggerConfig] = None):
        """Initialize with optional custom configuration."""

    def should_train(self, config_key: str) -> TriggerDecision:
        """
        Evaluate whether training should run for a config.

        Args:
            config_key: Board/player config (e.g., "square8_2p")

        Returns:
            TriggerDecision with should_train, reason, priority
        """

    def update_config_state(
        self,
        config_key: str,
        games_count: int,
        win_rate: float,
        model_count: int,
        current_elo: float = 1500.0,
    ) -> None:
        """
        Update the state for a config.

        Args:
            config_key: Board/player config
            games_count: Total games available
            win_rate: Current win rate (0.0-1.0)
            model_count: Number of trained models
            current_elo: Current model Elo rating
        """

    def record_training_start(self, config_key: str) -> None:
        """Record that training has started for a config."""

    def record_training_complete(
        self,
        config_key: str,
        games_used: int,
    ) -> None:
        """Record that training completed for a config."""

    def get_all_decisions(self) -> Dict[str, TriggerDecision]:
        """Get training decisions for all tracked configs."""

    def get_priority_queue(self) -> List[Tuple[str, float]]:
        """Get configs sorted by training priority (highest first)."""
```

### TriggerDecision Dataclass

```python
@dataclass
class TriggerDecision:
    """Result of training trigger evaluation."""
    should_train: bool           # Whether training should run
    reason: str                  # Human-readable reason
    signal_scores: Dict[str, float]  # Individual signal scores
    config_key: str              # Config being evaluated
    priority: float              # Combined priority score
```

### ConfigState Dataclass

```python
@dataclass
class ConfigState:
    """State for a single board/player configuration."""
    config_key: str
    games_since_training: int = 0
    last_training_time: float = 0
    last_training_games: int = 0
    model_count: int = 0
    current_elo: float = 1500.0
    win_rate: float = 0.5
    win_rate_trend: float = 0.0
```

## Integration Examples

### With Unified AI Loop

```python
from app.training.training_triggers import TrainingTriggers

triggers = TrainingTriggers()

# Main loop
while running:
    for config_key in all_configs:
        # Update state from database
        stats = get_config_stats(config_key)
        triggers.update_config_state(
            config_key,
            games_count=stats.total_games,
            win_rate=stats.win_rate,
            model_count=stats.model_count,
        )

        # Check if training needed
        decision = triggers.should_train(config_key)
        if decision.should_train:
            triggers.record_training_start(config_key)
            run_training(config_key)
            triggers.record_training_complete(config_key, games_used)

    time.sleep(60)
```

### Priority Queue Selection

```python
triggers = TrainingTriggers()

# Get all configs sorted by priority
queue = triggers.get_priority_queue()
# [("hexagonal_2p", 2.1), ("square8_3p", 1.5), ("square8_2p", 0.8), ...]

# Train top priority config
if queue:
    top_config, priority = queue[0]
    decision = triggers.should_train(top_config)
    if decision.should_train:
        run_training(top_config)
```

### With Curriculum Feedback

```python
from app.training.training_triggers import TrainingTriggers
from app.training.curriculum_feedback import get_curriculum_feedback

triggers = TrainingTriggers()
curriculum = get_curriculum_feedback()

# Update state with curriculum metrics
metrics = curriculum.get_config_metrics("square8_2p")
triggers.update_config_state(
    "square8_2p",
    games_count=metrics.games_total,
    win_rate=metrics.recent_win_rate,
    model_count=metrics.model_count,
)
```

## Signal Score Examples

### High Priority (Training Needed)

```python
decision = triggers.should_train("hexagonal_2p")
# TriggerDecision(
#     should_train=True,
#     reason="High priority: data_freshness (0.95), regression (0.40)",
#     signal_scores={
#         "data_freshness": 0.95,
#         "model_staleness": 0.30,
#         "performance_regression": 0.40,
#     },
#     config_key="hexagonal_2p",
#     priority=1.85,
# )
```

### Low Priority (No Training)

```python
decision = triggers.should_train("square8_2p")
# TriggerDecision(
#     should_train=False,
#     reason="Below threshold: priority=0.45",
#     signal_scores={
#         "data_freshness": 0.20,
#         "model_staleness": 0.15,
#         "performance_regression": 0.0,
#     },
#     config_key="square8_2p",
#     priority=0.45,
# )
```

### Bootstrap Config

```python
decision = triggers.should_train("hex8_4p")
# TriggerDecision(
#     should_train=True,
#     reason="Bootstrap: 0 models with 75 games available",
#     signal_scores={
#         "data_freshness": 1.0,  # 75 > 50 bootstrap threshold
#         "model_staleness": 1.0,
#         "performance_regression": 0.0,
#     },
#     config_key="hex8_4p",
#     priority=2.3,
# )
```

## Persistence

State is persisted to:

- **State file**: `data/training_triggers_state.json`

Contains:

- Per-config state (games count, last training time, model count)
- Last update timestamps

## Monitoring

### CLI Check

```bash
python -c "
from app.training.training_triggers import TrainingTriggers
triggers = TrainingTriggers()
for config, priority in triggers.get_priority_queue():
    decision = triggers.should_train(config)
    print(f'{config}: priority={priority:.2f} train={decision.should_train}')
"
```

### Prometheus Metrics

```
ringrift_trigger_priority{config="square8_2p"} 0.45
ringrift_trigger_freshness{config="square8_2p"} 0.20
ringrift_trigger_staleness{config="square8_2p"} 0.15
ringrift_trigger_regression{config="square8_2p"} 0.0
ringrift_trigger_should_train{config="square8_2p"} 0
```

## Comparison with Legacy System

| Aspect     | Legacy (8+ signals) | New (3 signals)    |
| ---------- | ------------------- | ------------------ |
| Signals    | 8+ overlapping      | 3 orthogonal       |
| Tuning     | Complex interaction | Independent        |
| Debugging  | Hard to trace       | Clear causality    |
| Bootstrap  | Implicit            | Explicit threshold |
| Regression | Mixed with others   | Dedicated signal   |

---

## See Also

- [CURRICULUM_FEEDBACK.md](CURRICULUM_FEEDBACK.md) - Curriculum weight adjustments
- [TRAINING_OPTIMIZATIONS.md](TRAINING_OPTIMIZATIONS.md) - Pipeline optimizations
- [UNIFIED_AI_LOOP.md](UNIFIED_AI_LOOP.md) - Main training loop
