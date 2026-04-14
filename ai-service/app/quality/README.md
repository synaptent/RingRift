# Quality Module

Unified quality scoring and monitoring for RingRift game data. This module provides the **single source of truth** for all quality-related computations across the AI service.

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
   - [UnifiedQualityScorer](#unifiedqualityscorer)
   - [GameQuality](#gamequality)
   - [QualityCategory](#qualitycategory)
3. [Thresholds](#thresholds)
4. [Quality Orchestrator](#quality-orchestrator)
5. [Scoring Components](#scoring-components)
6. [Usage Examples](#usage-examples)
7. [Configuration](#configuration)
8. [Integration](#integration)

---

## Overview

The quality module consolidates quality scoring from multiple deprecated implementations into a unified system:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Quality Scoring Architecture                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Game Data ─────► UnifiedQualityScorer ─────► GameQuality         │
│                           │                         │               │
│                    (compute scores)          (final result)         │
│                           │                         │               │
│              ┌────────────┴────────────┐           │               │
│              │                         │           │               │
│        Component Scores          Elo Weights      │               │
│        • Outcome (0-1)           • Avg Elo        │               │
│        • Length (0-1)            • Min Elo        │               │
│        • Balance (0-1)           • Max Elo        │               │
│        • Diversity (0-1)         • Elo Score      │               │
│        • Reputation (0-1)                         │               │
│              │                                     │               │
│              └──────────────┬──────────────────────┘               │
│                             ▼                                       │
│                      Final Score (0-1)                              │
│                             │                                       │
│                             ▼                                       │
│                    QualityCategory                                  │
│                    (excellent/good/adequate/poor/unusable)          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Consolidated From

This module replaces scattered implementations:

| Old Location                       | Now Use                               |
| ---------------------------------- | ------------------------------------- |
| `training/game_quality_scorer.py`  | `UnifiedQualityScorer`                |
| `distributed/quality_extractor.py` | `compute_sync_priority()`             |
| `training/streaming_pipeline.py`   | `compute_sample_weight()`             |
| `training/elo_weighting.py`        | Elo scoring in `UnifiedQualityScorer` |
| `distributed/unified_manifest.py`  | `GameQuality` dataclass               |

---

## Core Components

### UnifiedQualityScorer

The main entry point for all quality computations. Use the singleton instance for consistent behavior.

```python
from app.quality import (
    UnifiedQualityScorer,
    get_quality_scorer,
    compute_game_quality,
    compute_sample_weight,
    compute_sync_priority,
)

# Get singleton scorer
scorer = get_quality_scorer()

# Or create custom instance
scorer = UnifiedQualityScorer(
    outcome_weight=0.25,
    length_weight=0.15,
    balance_weight=0.20,
    diversity_weight=0.20,
    reputation_weight=0.10,
    elo_weight=0.10,
)
```

#### Methods

| Method                                          | Description                        |
| ----------------------------------------------- | ---------------------------------- |
| `compute_game_quality(game_data)`               | Full quality assessment for a game |
| `compute_sample_weight(quality, recency_hours)` | Training sample weight             |
| `compute_sync_priority(quality)`                | Data sync priority score           |
| `compute_elo_score(avg_elo, elo_diff)`          | Elo-based quality component        |

#### Convenience Functions

```python
from app.quality import (
    compute_game_quality,
    compute_game_quality_from_params,
    compute_sample_weight,
    compute_sync_priority,
    get_quality_category,
)

# Compute quality from game data dict
quality = compute_game_quality(game_data)

# Compute from explicit parameters
quality = compute_game_quality_from_params(
    game_id="abc123",
    winner_idx=0,
    total_moves=50,
    placement_moves=20,
    main_moves=30,
    source="gumbel_mcts",
    player_elos=[1600, 1550],
)

# Get sample weight for training
weight = compute_sample_weight(quality, recency_hours=2.0)

# Get sync priority
priority = compute_sync_priority(quality)

# Get category from score
category = get_quality_category(0.75)  # QualityCategory.GOOD
```

---

### GameQuality

Complete quality assessment result for a game.

```python
from app.quality import GameQuality

@dataclass
class GameQuality:
    game_id: str

    # Component scores (0-1 each)
    outcome_score: float = 0.0
    length_score: float = 0.0
    phase_balance_score: float = 0.0
    diversity_score: float = 0.0
    source_reputation_score: float = 0.0

    # Elo-based metrics
    avg_player_elo: float = 1500.0
    min_player_elo: float = 1500.0
    max_player_elo: float = 1500.0
    elo_difference: float = 0.0
    elo_score: float = 0.5

    # Game metadata
    total_moves: int = 0
    source: str = ""
    timestamp: float = 0.0

    # Final computed values
    final_score: float = 0.0
    category: QualityCategory = QualityCategory.ADEQUATE

    # Derived metrics
    sample_weight: float = 1.0
    sync_priority: float = 0.5
```

#### Usage

```python
quality = scorer.compute_game_quality(game_data)

print(f"Game: {quality.game_id}")
print(f"Final Score: {quality.final_score:.2f}")
print(f"Category: {quality.category.value}")
print(f"Sample Weight: {quality.sample_weight:.2f}")

# Component breakdown
print(f"Outcome: {quality.outcome_score:.2f}")
print(f"Length: {quality.length_score:.2f}")
print(f"Balance: {quality.phase_balance_score:.2f}")
print(f"Diversity: {quality.diversity_score:.2f}")
print(f"Elo: {quality.elo_score:.2f}")
```

---

### QualityCategory

Categorical quality classification.

```python
from app.quality import QualityCategory

class QualityCategory(str, Enum):
    EXCELLENT = "excellent"  # 0.85+
    GOOD = "good"            # 0.70-0.85
    ADEQUATE = "adequate"    # 0.50-0.70
    POOR = "poor"            # 0.30-0.50
    UNUSABLE = "unusable"    # <0.30
```

#### Category Thresholds

| Category    | Score Range | Use Case                |
| ----------- | ----------- | ----------------------- |
| `EXCELLENT` | >= 0.85     | Priority training, sync |
| `GOOD`      | 0.70-0.85   | Normal training         |
| `ADEQUATE`  | 0.50-0.70   | Low-weight training     |
| `POOR`      | 0.30-0.50   | Excluded from training  |
| `UNUSABLE`  | < 0.30      | Discarded               |

```python
# Get category from score
category = QualityCategory.from_score(0.75)  # GOOD

# Compare categories
if quality.category == QualityCategory.EXCELLENT:
    priority_sync(game)
```

---

## Thresholds

Quality thresholds for filtering and prioritization.

```python
from app.quality import (
    # Constants
    MIN_QUALITY_FOR_TRAINING,     # 0.3
    MIN_QUALITY_FOR_PRIORITY_SYNC,  # 0.5
    HIGH_QUALITY_THRESHOLD,       # 0.7

    # Helper functions
    is_training_worthy,
    is_priority_sync_worthy,
    is_high_quality,

    # Threshold container
    QualityThresholds,
    get_quality_thresholds,
)
```

#### Usage

```python
from app.quality import (
    is_training_worthy,
    is_high_quality,
    get_quality_thresholds,
)

score = 0.65

# Check thresholds
if is_training_worthy(score):
    include_in_training(game)

if is_high_quality(score):
    priority_sync(game)

# Get all thresholds
thresholds = get_quality_thresholds()
print(f"Training min: {thresholds.min_quality_for_training}")
print(f"Priority min: {thresholds.min_quality_for_priority_sync}")
print(f"High quality: {thresholds.high_quality_threshold}")

# Use threshold methods
if thresholds.is_high_quality(score):
    ...
```

---

## Quality Orchestrator

Centralized monitoring of quality events across the system.

**File**: `data_quality_orchestrator.py`

```python
from app.quality.data_quality_orchestrator import (
    DataQualityOrchestrator,
    wire_quality_events,
    get_quality_orchestrator,
    ConfigQualityState,
    QualityStats,
    QualityLevel,
)
```

### Event Subscriptions

The orchestrator subscribes to quality-related events:

| Event                          | Description                           |
| ------------------------------ | ------------------------------------- |
| `QUALITY_SCORE_UPDATED`        | Individual game quality computed      |
| `QUALITY_DISTRIBUTION_CHANGED` | Quality distribution shift detected   |
| `HIGH_QUALITY_DATA_AVAILABLE`  | Enough high-quality data for training |
| `LOW_QUALITY_DATA_WARNING`     | Quality degradation detected          |
| `DATA_QUALITY_ALERT`           | Critical quality issue                |

### Usage

```python
from app.quality.data_quality_orchestrator import (
    wire_quality_events,
    get_quality_orchestrator,
)

# Wire events and get orchestrator
orchestrator = wire_quality_events()

# Get overall status
status = orchestrator.get_status()
print(f"Configs tracked: {status['configs_tracked']}")
print(f"Ready for training: {status['configs_ready_for_training']}")

# Get config-specific quality
config_quality = orchestrator.get_config_quality("square8_2p")
print(f"Avg quality: {config_quality.avg_quality_score:.2f}")
print(f"Total games: {config_quality.total_games}")
print(f"Level: {config_quality.quality_level.value}")

# Check if config ready for training
if orchestrator.is_ready_for_training("hex8_2p"):
    trigger_training("hex8_2p")
```

### Quality Levels

```python
from app.quality.data_quality_orchestrator import QualityLevel

class QualityLevel(Enum):
    EXCELLENT = "excellent"  # >= 0.9
    GOOD = "good"            # >= 0.7
    ADEQUATE = "adequate"    # >= 0.5
    POOR = "poor"            # >= 0.3
    CRITICAL = "critical"    # < 0.3
```

---

## Scoring Components

### Outcome Score

Based on game result:

- Decisive win/loss: 1.0
- Draw: 0.7
- Resignation: 0.5
- Timeout/forfeit: 0.3

### Length Score

Based on game duration:

- Optimal range (40-100 moves): 1.0
- Too short (<20 moves): 0.3-0.7
- Too long (>200 moves): 0.5-0.8

### Phase Balance Score

Based on move distribution across phases:

- Balanced placement/main: 1.0
- Heavy placement bias: 0.6-0.8
- Heavy main bias: 0.6-0.8

### Diversity Score

Based on move variety:

- Many unique positions: 1.0
- Repetitive patterns: 0.4-0.6

### Source Reputation Score

Based on game source:

| Source        | Score |
| ------------- | ----- |
| `gumbel_mcts` | 1.0   |
| `nnue_guided` | 0.9   |
| `mcts`        | 0.85  |
| `policy_only` | 0.7   |
| `heuristic`   | 0.5   |
| `random`      | 0.2   |

### Elo Score

Based on player ratings:

- High avg Elo (>1600): Higher score
- Low Elo difference: Higher score (balanced match)
- Uses sigmoid normalization

---

## Usage Examples

### Batch Quality Scoring

```python
from app.quality import get_quality_scorer, is_training_worthy

scorer = get_quality_scorer()

# Score batch of games
high_quality_games = []
for game in games:
    quality = scorer.compute_game_quality(game)

    if is_training_worthy(quality.final_score):
        high_quality_games.append({
            "game": game,
            "quality": quality,
            "weight": quality.sample_weight,
        })

print(f"High quality: {len(high_quality_games)}/{len(games)}")
```

### Training Data Weighting

```python
from app.quality import compute_sample_weight, get_quality_scorer

scorer = get_quality_scorer()

# Weight samples for training
weighted_samples = []
for game, quality in scored_games:
    # Compute recency-adjusted weight
    hours_old = (time.time() - game.timestamp) / 3600
    weight = compute_sample_weight(quality, recency_hours=hours_old)

    weighted_samples.append((game, weight))

# Use weights in training
for game, weight in weighted_samples:
    loss = compute_loss(game)
    weighted_loss = loss * weight
```

### Sync Prioritization

```python
from app.quality import compute_sync_priority, get_quality_scorer

scorer = get_quality_scorer()

# Prioritize games for sync
sync_queue = []
for game in pending_games:
    quality = scorer.compute_game_quality(game)
    priority = compute_sync_priority(quality)

    sync_queue.append((game.id, priority))

# Sort by priority
sync_queue.sort(key=lambda x: x[1], reverse=True)

# Sync highest priority first
for game_id, priority in sync_queue[:100]:
    sync_game(game_id)
```

### Quality Monitoring

```python
from app.quality.data_quality_orchestrator import (
    wire_quality_events,
)

# Set up monitoring
orchestrator = wire_quality_events()

# Periodic check
def check_quality_health():
    status = orchestrator.get_status()

    if status["configs_with_warnings"] > 0:
        logger.warning(f"Quality warnings: {status['configs_with_warnings']}")

    for config_key in status["configs_tracked"]:
        config_state = orchestrator.get_config_quality(config_key)

        if config_state.quality_level == QualityLevel.CRITICAL:
            alert(f"Critical quality for {config_key}")
        elif config_state.quality_trend < -0.1:
            warn(f"Quality degrading for {config_key}")
```

---

## Configuration

### Environment Variables

| Variable                            | Default | Description                  |
| ----------------------------------- | ------- | ---------------------------- |
| `RINGRIFT_MIN_QUALITY_FOR_TRAINING` | 0.3     | Minimum quality for training |
| `RINGRIFT_HIGH_QUALITY_THRESHOLD`   | 0.7     | High quality threshold       |
| `RINGRIFT_QUALITY_RECENCY_DECAY`    | 0.1     | Recency decay rate           |

### Custom Weights

```python
from app.quality import UnifiedQualityScorer

# Custom weight configuration
scorer = UnifiedQualityScorer(
    outcome_weight=0.30,    # More weight on outcome
    length_weight=0.10,     # Less weight on length
    balance_weight=0.20,
    diversity_weight=0.15,
    reputation_weight=0.15,
    elo_weight=0.10,
)
```

---

## Integration

### Event System

```python
from app.coordination.event_router import get_router

router = get_router()

# Quality events are automatically emitted
# when using UnifiedQualityScorer with event integration

@router.on("QUALITY_SCORE_UPDATED")
async def handle_quality_update(event):
    game_id = event.data["game_id"]
    score = event.data["final_score"]
    logger.info(f"Quality updated: {game_id} = {score:.2f}")
```

### Training Pipeline

```python
from app.quality import get_quality_scorer, is_training_worthy
from app.training.optimized_pipeline import get_optimized_pipeline

scorer = get_quality_scorer()
pipeline = get_optimized_pipeline()

quality = scorer.compute_game_quality(game_data)
if is_training_worthy(quality.final_score):
    should_train, reason = pipeline.should_train(
        "square8_2p",
        games_since_training=800,
        current_elo=1600.0,
    )
    if should_train:
        result = pipeline.run_training(
            "square8_2p",
            db_paths=["data/games/canonical_square8_2p.db"],
        )
        print(result.message)
    else:
        print(f"Training deferred: {reason}")
```

Use `app.training.train_loop.run_training_loop` for the full selfplay -> train -> evaluate loop. Use
`get_optimized_pipeline()` when integrating quality gates with direct export and training decisions.

---

## Module Reference

| File                           | Lines | Description              |
| ------------------------------ | ----- | ------------------------ |
| `unified_quality.py`           | 750   | Core quality scorer      |
| `data_quality_orchestrator.py` | 550   | Quality event monitoring |
| `thresholds.py`                | 90    | Threshold constants      |

---

## See Also

- `app/training/README.md` - Training pipeline integration
- `app/distributed/README.md` - Sync prioritization
- `app/config/thresholds.py` - Canonical threshold values
- `app/coordination/README.md` - Event system

---

_Last updated: December 2025_
