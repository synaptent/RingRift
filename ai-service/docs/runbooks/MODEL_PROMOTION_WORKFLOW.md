# Model Promotion Workflow Runbook

**Last Updated**: December 28, 2025
**Version**: Wave 7

## Overview

Model promotion is the process of transitioning a newly trained model to become the canonical model for its configuration. This involves gauntlet evaluation, comparison against baselines, and cluster-wide distribution.

## Workflow Diagram

```
Training Completed
        |
        v
TRAINING_COMPLETED event emitted
        |
        v
FeedbackLoopController triggers gauntlet
        |
        v
GameGauntlet runs evaluation
        |
        v
EVALUATION_COMPLETED event emitted
        |
        v
PromotionController evaluates results
        |
  [Pass thresholds?]
   /          \
 Yes           No
  |             |
  v             v
MODEL_PROMOTED   Continue training
event emitted    (no promotion)
  |
  v
UnifiedDistributionDaemon distributes model
        |
        v
Model available cluster-wide
```

## Key Components

| Component                   | Location                                          | Purpose                             |
| --------------------------- | ------------------------------------------------- | ----------------------------------- |
| `PromotionController`       | `app/training/promotion_controller.py`            | Evaluates models against thresholds |
| `GameGauntlet`              | `app/training/game_gauntlet.py`                   | Runs evaluation games vs baselines  |
| `UnifiedDistributionDaemon` | `app/coordination/unified_distribution_daemon.py` | Distributes promoted models         |
| `FeedbackLoopController`    | `app/coordination/feedback_loop_controller.py`    | Triggers gauntlet after training    |

## Promotion Thresholds

Default thresholds (from `app/config/thresholds.py`):

| Baseline       | Win Rate Required |
| -------------- | ----------------- |
| Random         | 85%               |
| Heuristic      | 60%               |
| Previous Model | 50% (optional)    |

Environment variable overrides:

- `RINGRIFT_PROMOTION_RANDOM_WIN_RATE=0.85`
- `RINGRIFT_PROMOTION_HEURISTIC_WIN_RATE=0.60`
- `RINGRIFT_PROMOTION_VS_PREVIOUS_WIN_RATE=0.50`

## Events

### Emitters

| Event                    | Emitter             | When                         |
| ------------------------ | ------------------- | ---------------------------- |
| `TRAINING_COMPLETED`     | TrainingCoordinator | Training run finishes        |
| `EVALUATION_COMPLETED`   | GameGauntlet        | Gauntlet evaluation finishes |
| `MODEL_PROMOTED`         | PromotionController | Model passes all thresholds  |
| `MODEL_PROMOTION_FAILED` | PromotionController | Model fails thresholds       |

### Subscribers

| Event                  | Subscriber                | Action                     |
| ---------------------- | ------------------------- | -------------------------- |
| `TRAINING_COMPLETED`   | FeedbackLoopController    | Triggers gauntlet          |
| `EVALUATION_COMPLETED` | PromotionController       | Evaluates results          |
| `MODEL_PROMOTED`       | UnifiedDistributionDaemon | Distributes to cluster     |
| `MODEL_PROMOTED`       | CurriculumIntegration     | Updates curriculum weights |

## Manual Promotion

### Using auto_promote.py

```bash
# Evaluate and optionally promote a model
python scripts/auto_promote.py --gauntlet \
  --model models/my_model.pth \
  --board-type hex8 --num-players 2 \
  --games 50 --sync-to-cluster

# Dry run (evaluate without promoting)
python scripts/auto_promote.py --gauntlet \
  --model models/my_model.pth \
  --board-type hex8 --num-players 2 \
  --games 50 --dry-run

# Force promotion (skip gauntlet)
python scripts/auto_promote.py --force \
  --model models/my_model.pth \
  --board-type hex8 --num-players 2
```

### Via P2P Admin Endpoint

```bash
# Trigger promotion evaluation
curl -X POST http://localhost:8770/admin/promote \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "models/my_model.pth",
    "board_type": "hex8",
    "num_players": 2,
    "gauntlet_games": 50
  }'

# Check promotion status
curl -s http://localhost:8770/status | jq '.promotion_status'
```

### Direct Programmatic API

```python
from app.training.promotion_controller import PromotionController

controller = PromotionController()

# Run gauntlet and promote if passes
result = await controller.evaluate_and_promote(
    model_path="models/my_model.pth",
    board_type="hex8",
    num_players=2,
    gauntlet_games=50,
)

if result.promoted:
    print(f"Model promoted to: {result.canonical_path}")
else:
    print(f"Promotion failed: {result.failure_reason}")
```

## Model Distribution

After promotion, models are distributed via `UnifiedDistributionDaemon`:

### Distribution Status

```bash
# Check distribution status
curl -s http://localhost:8770/status | jq '.model_distribution'

# Check which nodes have the model
python -c "
from app.coordination.unified_distribution_daemon import check_model_availability
availability = check_model_availability('canonical_hex8_2p.pth')
for node, has_model in availability.items():
    print(f'{node}: {\"OK\" if has_model else \"MISSING\"}')
"
```

### Manual Distribution

```bash
# Force distribution of a model
curl -X POST http://localhost:8770/admin/distribute-model \
  -H "Content-Type: application/json" \
  -d '{"model_path": "models/canonical_hex8_2p.pth"}'

# Wait for distribution to complete
python -c "
from app.coordination.unified_distribution_daemon import wait_for_model_distribution
await wait_for_model_distribution('canonical_hex8_2p.pth', timeout=300)
print('Distribution complete')
"
```

## Troubleshooting

### 1. Model Not Promoting After Training

**Symptoms**: Training completes but no MODEL_PROMOTED event

**Diagnosis**:

```bash
# Check if EVALUATION_COMPLETED was emitted
curl -s http://localhost:8770/status | jq '.recent_events | map(select(.type | contains("EVAL")))'

# Check gauntlet results
grep -i "gauntlet" logs/coordination.log | tail -20

# Check promotion controller status
python -c "
from app.training.promotion_controller import PromotionController
ctrl = PromotionController.get_instance()
print(f'Last evaluation: {ctrl.last_evaluation}')
"
```

**Common Fixes**:

1. Model failed thresholds - check win rates in logs
2. Gauntlet not triggered - verify FeedbackLoopController subscription
3. Event not reaching PromotionController - check subscription wiring

### 2. Distribution Stuck

**Symptoms**: MODEL_PROMOTED emitted but nodes don't receive model

**Diagnosis**:

```bash
# Check distribution daemon status
curl -s http://localhost:8770/status | jq '.daemons.MODEL_DISTRIBUTION'

# Check for transfer errors
grep -i "distribution" logs/coordination.log | grep -i "error\|fail" | tail -10

# Check node connectivity
for node in $(curl -s http://localhost:8770/status | jq -r '.alive_peers[]'); do
  echo "Checking $node..."
  curl -s --connect-timeout 5 "http://$node:8780/health" || echo "UNREACHABLE"
done
```

**Common Fixes**:

1. Network connectivity - verify SSH/P2P access to target nodes
2. Disk space - check disk usage on target nodes
3. Circuit breaker open - check transport status

### 3. Wrong Model Promoted

**Symptoms**: Canonical model not the expected one

**Diagnosis**:

```bash
# Check canonical symlink
ls -la models/ | grep canonical

# Check model metadata
python -c "
import torch
model = torch.load('models/canonical_hex8_2p.pth', weights_only=False)
print(f'Version: {model.get(\"version\")}')
print(f'Trained: {model.get(\"trained_at\")}')
print(f'Elo: {model.get(\"elo\")}')
"
```

**Fix**: Restore correct model

```bash
# Restore from backup
cp models/backup/canonical_hex8_2p.pth.bak models/canonical_hex8_2p.pth

# Update symlink
ln -sf canonical_hex8_2p.pth models/ringrift_best_hex8_2p.pth

# Trigger redistribution
curl -X POST http://localhost:8770/admin/distribute-model \
  -d '{"model_path": "models/canonical_hex8_2p.pth", "force": true}'
```

### 4. Gauntlet Evaluation Failing

**Symptoms**: EVALUATION_COMPLETED not emitted after TRAINING_COMPLETED

**Diagnosis**:

```bash
# Check if gauntlet was triggered
grep "run_gauntlet" logs/coordination.log | tail -10

# Check for exceptions
grep -i "gauntlet.*exception\|gauntlet.*error" logs/coordination.log | tail -10

# Check GPU availability
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

**Common Fixes**:

1. OOM during gauntlet - reduce parallel games or use smaller batch
2. Missing model file - verify path exists
3. Import errors - check module dependencies

## Environment Variables

| Variable                                | Default | Description                      |
| --------------------------------------- | ------- | -------------------------------- |
| `RINGRIFT_AUTO_PROMOTE`                 | true    | Enable automatic promotion       |
| `RINGRIFT_GAUNTLET_GAMES`               | 50      | Games per opponent in gauntlet   |
| `RINGRIFT_PROMOTION_RANDOM_WIN_RATE`    | 0.85    | Required win rate vs random      |
| `RINGRIFT_PROMOTION_HEURISTIC_WIN_RATE` | 0.60    | Required win rate vs heuristic   |
| `RINGRIFT_DISTRIBUTION_TIMEOUT`         | 300     | Seconds to wait for distribution |
| `RINGRIFT_SKIP_GAUNTLET`                | false   | Skip gauntlet (force promote)    |

## See Also

- [FEEDBACK_LOOP_TROUBLESHOOTING.md](FEEDBACK_LOOP_TROUBLESHOOTING.md) - Feedback loop issues
- [TRAINING_LOOP_STALLED.md](TRAINING_LOOP_STALLED.md) - Training issues
- [COORDINATION_EVENT_SYSTEM.md](COORDINATION_EVENT_SYSTEM.md) - Event system reference
- [DAEMON_MANAGER_OPERATIONS.md](DAEMON_MANAGER_OPERATIONS.md) - Daemon lifecycle
