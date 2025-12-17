# Curriculum Feedback System

> **Last Updated**: 2025-12-17
> **Status**: Active
> **Location**: `app/training/curriculum_feedback.py`

This document describes the curriculum feedback loop that dynamically adjusts training weights based on selfplay performance.

## Overview

The curriculum feedback system closes the loop between selfplay results and training priorities:

1. **Track selfplay metrics** (win rates, game counts) per config
2. **Feed back to curriculum weights** more frequently than hourly
3. **Adjust selfplay allocation** based on model performance

This creates a responsive system where:

- Weak configs get more training attention
- Strong configs get less training (resources reallocated)
- Metrics update in near real-time

## Quick Start

```python
from app.training.curriculum_feedback import CurriculumFeedback, get_curriculum_feedback

# Get singleton instance
feedback = get_curriculum_feedback()

# Record selfplay results
feedback.record_game("square8_2p", winner=1, model_elo=1650)
feedback.record_game("hexagonal_2p", winner=-1, model_elo=1400)

# Get updated curriculum weights
weights = feedback.get_curriculum_weights()
# {"square8_2p": 0.8, "hexagonal_2p": 1.2, ...}

# Export weights for P2P orchestrator
feedback.export_weights_json("curriculum_weights.json")
```

## Configuration

### Constructor Parameters

| Parameter          | Type  | Default | Description                           |
| ------------------ | ----- | ------- | ------------------------------------- |
| `lookback_minutes` | int   | 30      | Time window for recent game metrics   |
| `weight_min`       | float | 0.5     | Minimum curriculum weight             |
| `weight_max`       | float | 2.0     | Maximum curriculum weight             |
| `target_win_rate`  | float | 0.55    | Target win rate for balanced training |

### Example Configuration

```python
feedback = CurriculumFeedback(
    lookback_minutes=30,
    weight_min=0.5,
    weight_max=2.0,
    target_win_rate=0.55,
)
```

## Weight Calculation

Weights range from 0.5 (de-prioritize) to 2.0 (high priority) based on several factors:

| Factor               | Effect                  |
| -------------------- | ----------------------- |
| Low win rate (<55%)  | +0.4 weight             |
| No trained models    | +0.5 weight (bootstrap) |
| Elo regression       | +0.2 weight             |
| High win rate (>70%) | -0.3 weight             |
| Stale config (>6h)   | +0.1 weight             |

### Weight Formula

```python
weight = 1.0  # Base weight

# Win rate adjustment
if recent_win_rate < target_win_rate:
    weight += 0.4 * (target_win_rate - recent_win_rate) / target_win_rate

# Bootstrap bonus for new configs
if model_count == 0:
    weight += 0.5

# Elo regression penalty
if elo_trend < 0:
    weight += 0.2 * abs(elo_trend) / 50

# High performer reduction
if recent_win_rate > 0.70:
    weight -= 0.3

# Staleness bonus
if hours_since_last_game > 6:
    weight += 0.1

# Clamp to range
weight = max(weight_min, min(weight_max, weight))
```

## API Reference

### CurriculumFeedback Class

```python
class CurriculumFeedback:
    """Manages curriculum feedback loop with real-time metrics."""

    def __init__(
        self,
        lookback_minutes: int = 30,
        weight_min: float = 0.5,
        weight_max: float = 2.0,
        target_win_rate: float = 0.55,
    ):
        """Initialize curriculum feedback system."""

    def record_game(
        self,
        config_key: str,
        winner: int,
        model_elo: float = 1500.0,
        opponent_type: str = "baseline",
    ) -> None:
        """
        Record a selfplay game result.

        Args:
            config_key: Board/player config (e.g., "square8_2p")
            winner: 1 = model won, -1 = model lost, 0 = draw
            model_elo: Current model Elo rating
            opponent_type: Type of opponent (baseline, selfplay, etc.)
        """

    def record_selfplay_game(
        self,
        config_key: str,
        winner: int,
        model_elo: float = 1500.0,
    ) -> None:
        """Alias for record_game with opponent_type='selfplay'."""

    def get_curriculum_weights(self) -> Dict[str, float]:
        """
        Get current curriculum weights for all configs.

        Returns:
            Dictionary mapping config_key to weight (0.5-2.0)
        """

    def get_config_metrics(self, config_key: str) -> ConfigMetrics:
        """
        Get detailed metrics for a specific config.

        Returns:
            ConfigMetrics dataclass with win rates, game counts, etc.
        """

    def export_weights_json(self, path: str) -> None:
        """Export weights to JSON file for P2P orchestrator."""

    def update_model_count(self, config_key: str, count: int) -> None:
        """Update the trained model count for a config."""

    def update_training_time(self, config_key: str) -> None:
        """Record that training just completed for a config."""
```

### ConfigMetrics Dataclass

```python
@dataclass
class ConfigMetrics:
    """Metrics for a single config."""
    games_total: int = 0
    games_recent: int = 0       # In lookback window
    wins_recent: int = 0
    losses_recent: int = 0
    draws_recent: int = 0
    avg_elo: float = 1500.0
    win_rate: float = 0.5
    elo_trend: float = 0.0      # Positive = improving
    last_game_time: float = 0
    last_training_time: float = 0
    model_count: int = 0

    @property
    def recent_win_rate(self) -> float:
        """Win rate over recent games only."""
```

## Integration

### With Unified AI Loop

The unified loop queries curriculum weights during training decisions:

```python
from app.training.curriculum_feedback import get_curriculum_feedback

feedback = get_curriculum_feedback()
weights = feedback.get_curriculum_weights()

# Select config with highest weight for training
priority_config = max(weights.items(), key=lambda x: x[1])[0]
```

### With P2P Orchestrator

The P2P orchestrator uses exported weights to balance selfplay:

```python
# Export weights periodically
feedback.export_weights_json("data/curriculum_weights.json")

# P2P orchestrator reads and applies weights
with open("data/curriculum_weights.json") as f:
    weights = json.load(f)

# Allocate selfplay proportionally
for config, weight in weights.items():
    workers = int(base_workers * weight)
    spawn_selfplay(config, workers)
```

### With Training Pipeline

```python
from app.training.curriculum_feedback import get_curriculum_feedback

feedback = get_curriculum_feedback()

# Before training
metrics = feedback.get_config_metrics("square8_2p")
print(f"Win rate: {metrics.recent_win_rate:.1%}")
print(f"Elo trend: {metrics.elo_trend:+.1f}")

# After training completes
feedback.update_training_time("square8_2p")
feedback.update_model_count("square8_2p", new_model_count)
```

## Persistence

Curriculum state is persisted to disk at:

- **State file**: `data/curriculum_feedback_state.json`
- **Weights export**: `data/curriculum_weights.json`

The state file contains:

- Game history per config (circular buffer)
- Computed metrics
- Last update timestamps

## Monitoring

### Get All Metrics

```python
feedback = get_curriculum_feedback()

for config_key in feedback.get_all_config_keys():
    metrics = feedback.get_config_metrics(config_key)
    weight = feedback.get_curriculum_weights()[config_key]

    print(f"{config_key}:")
    print(f"  Weight: {weight:.2f}")
    print(f"  Win rate: {metrics.recent_win_rate:.1%}")
    print(f"  Elo trend: {metrics.elo_trend:+.1f}")
    print(f"  Games (recent): {metrics.games_recent}")
```

### Prometheus Metrics

The curriculum feedback system exposes Prometheus metrics:

```
ringrift_curriculum_weight{config="square8_2p"} 1.2
ringrift_curriculum_win_rate{config="square8_2p"} 0.55
ringrift_curriculum_elo_trend{config="square8_2p"} 15.0
ringrift_curriculum_games_recent{config="square8_2p"} 150
```

## Best Practices

1. **Record all games**: Call `record_game()` for every selfplay game, not just samples
2. **Update model counts**: Keep model counts accurate for bootstrap detection
3. **Periodic export**: Export weights every 5-10 minutes for P2P sync
4. **Monitor stale configs**: Check for configs with 0 recent games

## Troubleshooting

### Weight Always at 1.0

- Check that games are being recorded
- Verify lookback window isn't too short
- Ensure model counts are updated

### Rapid Weight Oscillation

- Increase `lookback_minutes` for more stable weights
- Add smoothing (EMA) to weight updates
- Check for inconsistent game recording

---

## See Also

- [TRAINING_TRIGGERS.md](TRAINING_TRIGGERS.md) - Training decision triggers
- [TRAINING_OPTIMIZATIONS.md](TRAINING_OPTIMIZATIONS.md) - Pipeline optimizations
- [UNIFIED_AI_LOOP.md](UNIFIED_AI_LOOP.md) - Main training loop
