# Tier Promotion System

The tier promotion system manages model progression through difficulty tiers (D1-D9) based on evaluation performance. It ensures models meet quality gates before being promoted to serve harder difficulty levels.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TIER PROMOTION FLOW                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Training → Candidate → Evaluation → Gate → Promotion/Reject │
│     │          │            │          │          │          │
│     ▼          ▼            ▼          ▼          ▼          │
│  Model     Registry     Tournament   Criteria   Ladder      │
│  Output    Tracking     Results      Check      Update      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Core Modules

### tier_promotion_registry.py

Tracks tier promotion candidates and their status.

```python
from app.training.tier_promotion_registry import (
    load_square8_two_player_registry,
    save_square8_two_player_registry,
    get_current_ladder_model_for_tier,
)

# Load current registry
registry = load_square8_two_player_registry()

# Check tier assignment
current = get_current_ladder_model_for_tier("D4")
print(f"D4 model: {current['model_id']}")
```

**Registry Structure:**

```json
{
  "board": "square8",
  "num_players": 2,
  "tiers": {
    "D4": {
      "candidates": [
        {
          "model_id": "model_v3_20241215",
          "status": "pending",
          "submitted_at": "2024-12-15T10:00:00Z",
          "eval_results": null
        }
      ],
      "current_model": "model_v3_20241210"
    }
  }
}
```

### tier_eval_runner.py

Orchestrates tier evaluation tournaments.

```python
from app.training.tier_eval_runner import TierEvalRunner

runner = TierEvalRunner(
    board_type="square8",
    num_players=2,
    tier="D4",
)

# Run evaluation tournament
results = await runner.evaluate_candidate(
    candidate_model="model_v3_20241215",
    baseline_model="model_v3_20241210",
    games=100,
)

print(f"Win rate: {results.win_rate:.1%}")
print(f"Elo difference: {results.elo_diff:+.0f}")
```

### tier_eval_config.py

Configuration for tier evaluation criteria.

```python
from app.training.tier_eval_config import TierEvalConfig, get_tier_config

# Get config for specific tier
config = get_tier_config("D4")

print(f"Min win rate: {config.min_win_rate:.1%}")
print(f"Min games: {config.min_games}")
print(f"Elo threshold: {config.elo_threshold}")
```

**Default Tier Configurations:**

| Tier | Min Win Rate | Min Games | Elo Threshold |
| ---- | ------------ | --------- | ------------- |
| D1   | 55%          | 50        | +10           |
| D2   | 55%          | 50        | +10           |
| D3   | 55%          | 75        | +15           |
| D4   | 55%          | 100       | +20           |
| D5   | 57%          | 100       | +25           |
| D6   | 57%          | 150       | +30           |
| D7   | 58%          | 150       | +35           |
| D8   | 58%          | 200       | +40           |
| D9   | 60%          | 200       | +50           |

### tier_perf_benchmark.py

Benchmarks tier performance for monitoring.

```python
from app.training.tier_perf_benchmark import (
    benchmark_tier_performance,
    TierBenchmarkResult,
)

# Run benchmark
result = benchmark_tier_performance(
    tier="D4",
    games=50,
    timeout=300,
)

print(f"Games/second: {result.games_per_second:.2f}")
print(f"Average game length: {result.avg_game_length:.1f} moves")
```

## Promotion Workflow

### 1. Candidate Submission

After training completes, models are submitted as promotion candidates:

```python
from app.training.tier_promotion_registry import submit_candidate

submit_candidate(
    tier="D4",
    model_id="model_v3_20241215",
    model_path="/path/to/model.pt",
    training_metrics={
        "final_loss": 0.45,
        "epochs": 50,
        "training_games": 50000,
    },
)
```

### 2. Evaluation Tournament

The evaluation runner schedules tournaments against the current tier model:

```bash
# Manual evaluation
python scripts/run_tier_evaluation.py \
    --tier D4 \
    --candidate model_v3_20241215 \
    --games 100
```

### 3. Gate Decision

Based on evaluation results, promotion is gated:

```python
from app.training.tier_eval_config import check_promotion_gate

decision = check_promotion_gate(
    tier="D4",
    win_rate=0.58,
    elo_diff=25,
    games_played=100,
)

if decision.approved:
    print(f"Promotion approved: {decision.reason}")
else:
    print(f"Promotion rejected: {decision.reason}")
```

### 4. Ladder Update

On approval, the ladder configuration is updated:

```python
from app.training.tier_promotion_registry import promote_candidate

promote_candidate(
    tier="D4",
    model_id="model_v3_20241215",
)
# Updates config/ladder_config.py and registry
```

## Integration with Training Pipeline

The tier system integrates with the unified training loop:

```python
# In unified_ai_loop.py
from app.training.tier_promotion_registry import get_pending_candidates

# Check for candidates needing evaluation
pending = get_pending_candidates()
for candidate in pending:
    # Schedule evaluation
    await scheduler.schedule_tier_eval(candidate)
```

## Monitoring

### Registry Status

```bash
# View current tier assignments
python -c "
from app.training.tier_promotion_registry import load_square8_two_player_registry
import json
reg = load_square8_two_player_registry()
print(json.dumps(reg, indent=2))
"
```

### Evaluation Logs

Evaluation results are logged to:

- `data/tier_eval/results_{tier}_{timestamp}.json`
- `logs/tier_eval.log`

### Prometheus Metrics

```
# Tier promotion metrics
ringrift_tier_evaluations_total{tier="D4", result="promoted"} 5
ringrift_tier_evaluations_total{tier="D4", result="rejected"} 3
ringrift_tier_candidate_queue_size{tier="D4"} 2
ringrift_tier_current_elo{tier="D4"} 1650
```

## Configuration

### Environment Variables

| Variable            | Default | Description                  |
| ------------------- | ------- | ---------------------------- |
| `TIER_EVAL_GAMES`   | 100     | Default games per evaluation |
| `TIER_EVAL_TIMEOUT` | 3600    | Evaluation timeout (seconds) |
| `TIER_AUTO_PROMOTE` | false   | Auto-promote on gate pass    |

### Config Files

- `config/tier_candidate_registry.square8_2p.json` - Candidate tracking
- `config/ladder_config.py` - Live tier assignments
- `config/tier_eval_thresholds.yaml` - Custom gate thresholds

## Troubleshooting

### Evaluation Stuck

```bash
# Check active evaluations
python scripts/check_tier_eval_status.py

# Force timeout stale evaluations
python scripts/cleanup_tier_evals.py --timeout 7200
```

### Candidate Not Promoted

1. Check evaluation results in registry
2. Verify gate criteria in tier_eval_config.py
3. Review evaluation logs for errors

### Registry Corruption

```bash
# Backup and reset registry
cp config/tier_candidate_registry.square8_2p.json config/tier_registry_backup.json
python -c "
from app.training.tier_promotion_registry import save_square8_two_player_registry
save_square8_two_player_registry({'board': 'square8', 'num_players': 2, 'tiers': {}})
"
```

## Related Documentation

- [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md) - Training workflow
- [UNIFIED_AI_LOOP.md](UNIFIED_AI_LOOP.md) - Automated promotion in unified loop
- [AI_LADDER_PRODUCTION_RUNBOOK.md](../../docs/ai/AI_LADDER_PRODUCTION_RUNBOOK.md) - Difficulty ladder operations
