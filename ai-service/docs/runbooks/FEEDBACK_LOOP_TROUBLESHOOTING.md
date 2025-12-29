# Feedback Loop Troubleshooting Runbook

**Last Updated**: December 28, 2025
**Version**: Wave 7

## Overview

RingRift uses multiple feedback loops to improve AI model quality continuously. This runbook covers diagnosis and resolution of issues in the feedback loop infrastructure.

## Feedback Loop Architecture

```
Selfplay → Games → Export → Training → Gauntlet → Promotion → Distribution
    ↑                                                              |
    +-------------------------- Curriculum Feedback ---------------+
```

### Key Feedback Loops

| Loop                    | Components                                      | Purpose                                       |
| ----------------------- | ----------------------------------------------- | --------------------------------------------- |
| **Training Feedback**   | TrainingCoordinator → FeedbackLoopController    | Update hyperparameters based on loss curves   |
| **Gauntlet Feedback**   | GameGauntlet → GauntletFeedbackController       | Adjust curriculum based on evaluation results |
| **Curriculum Feedback** | CurriculumIntegration → SelfplayScheduler       | Rebalance training priorities                 |
| **Quality Feedback**    | QualityMonitorDaemon → DataPipelineOrchestrator | Adjust data quality thresholds                |

## Quick Diagnosis

```bash
# Check all feedback loop status
curl -s http://localhost:8770/status | python3 -c '
import sys, json
d = json.load(sys.stdin)
print("=== Feedback Loop Status ===")
for daemon in ["FEEDBACK_LOOP", "CURRICULUM_INTEGRATION", "QUALITY_MONITOR"]:
    info = d.get("daemons", {}).get(daemon, {})
    print(f"{daemon}: {info.get(\"state\", \"UNKNOWN\")} - last_run: {info.get(\"last_run\", \"never\")}")
'

# Check recent feedback events
curl -s http://localhost:8770/status | jq '.recent_events | map(select(.type | test("FEEDBACK|CURRICULUM|QUALITY")))'

# Check subscription wiring
python -c "
from app.coordination.event_router import get_event_bus
bus = get_event_bus()
subs = bus.list_subscribers()
feedback_events = [k for k in subs if 'feedback' in k.lower() or 'curriculum' in k.lower()]
for event in feedback_events:
    print(f'{event}: {len(subs[event])} subscriber(s)')
"
```

## Common Issues & Fixes

### 1. TRAINING_COMPLETED Not Triggering Gauntlet

**Symptoms**: Training completes but gauntlet never runs

**Diagnosis**:

```bash
# Check if event was emitted
grep "TRAINING_COMPLETED" logs/coordination.log | tail -5

# Check if FeedbackLoopController is subscribed
python -c "
from app.coordination.event_router import get_event_bus
bus = get_event_bus()
print('TRAINING_COMPLETED subscribers:', bus.list_subscribers().get('training_completed', []))
"

# Check FeedbackLoopController status
python -c "
from app.coordination.feedback_loop_controller import FeedbackLoopController
ctrl = FeedbackLoopController.get_instance()
print(f'Running: {ctrl._running}')
print(f'Last trigger: {ctrl.last_gauntlet_trigger}')
"
```

**Fixes**:

```python
# Manual subscription wiring
from app.coordination.feedback_loop_controller import FeedbackLoopController
from app.coordination.event_router import subscribe

ctrl = FeedbackLoopController.get_instance()
subscribe("training_completed", ctrl._on_training_completed)
```

### 2. Curriculum Weights Not Updating

**Symptoms**: Selfplay always runs same configs despite poor performance

**Diagnosis**:

```bash
# Check current curriculum weights
python -c "
from app.coordination.curriculum_integration import CurriculumIntegration
ci = CurriculumIntegration.get_instance()
weights = ci.get_current_weights()
for config, weight in weights.items():
    print(f'{config}: {weight:.3f}')
"

# Check CURRICULUM_REBALANCED events
grep "CURRICULUM_REBALANCED" logs/coordination.log | tail -10

# Check if evaluation results are being received
grep "EVALUATION_COMPLETED" logs/coordination.log | tail -5
```

**Fixes**:

```python
# Manual curriculum rebalance
from app.coordination.curriculum_integration import CurriculumIntegration

ci = CurriculumIntegration.get_instance()

# Force rebalance based on current data
await ci.rebalance_weights()

# Or manually set weights
ci.set_weights({
    "hex8_2p": 0.3,
    "hex8_4p": 0.2,
    "square8_2p": 0.25,
    "square8_4p": 0.25,
})
```

### 3. Quality Feedback Not Adjusting Thresholds

**Symptoms**: Low-quality data continues being used

**Diagnosis**:

```bash
# Check quality scores
python -c "
from app.training.data_quality import get_quality_scores
scores = get_quality_scores()
for db, score in scores.items():
    print(f'{db}: quality={score.quality:.2f}, freshness={score.freshness:.2f}')
"

# Check quality threshold settings
grep "QUALITY_THRESHOLD" logs/coordination.log | tail -5

# Check quality events
curl -s http://localhost:8770/status | jq '.recent_events | map(select(.type | contains("QUALITY")))'
```

**Fixes**:

```python
# Adjust quality thresholds
from app.config.thresholds import update_quality_thresholds

update_quality_thresholds(
    min_quality=0.5,  # Raise minimum quality
    freshness_hours=12,  # Require fresher data
)

# Or trigger quality check manually
from app.coordination.quality_monitor_daemon import QualityMonitorDaemon
qm = QualityMonitorDaemon.get_instance()
await qm._run_cycle()
```

### 4. Hyperparameter Feedback Not Working

**Symptoms**: Learning rate stays constant despite plateaus

**Diagnosis**:

```bash
# Check hyperparameter adjustment events
grep "HYPERPARAMETER_UPDATED" logs/coordination.log | tail -10

# Check regression detection
grep "LOSS_REGRESSION\|PLATEAU_DETECTED" logs/coordination.log | tail -10

# Check current hyperparameters
python -c "
from app.training.adaptive_controller import AdaptiveController
ctrl = AdaptiveController.get_instance()
print(f'Current LR: {ctrl.current_lr}')
print(f'LR schedule: {ctrl.lr_schedule}')
print(f'Plateau count: {ctrl.plateau_count}')
"
```

**Fixes**:

```python
# Trigger hyperparameter adjustment manually
from app.coordination.gauntlet_feedback_controller import GauntletFeedbackController

gfc = GauntletFeedbackController.get_instance()
await gfc.apply_hyperparameter_adjustment({
    "learning_rate": 0.0001,  # Reduce LR
    "reason": "plateau_detected",
})

# Or reset to defaults
from app.training.adaptive_controller import AdaptiveController
ctrl = AdaptiveController.get_instance()
ctrl.reset_to_defaults()
```

### 5. Event Handler Timeout

**Symptoms**: Feedback handlers not completing, DLQ growing

**Diagnosis**:

```bash
# Check handler timeouts
curl -s http://localhost:8770/status | jq '.handler_timeouts'

# Check DLQ size
python -c "
from app.coordination.dead_letter_queue import get_dead_letter_queue
dlq = get_dead_letter_queue()
print(f'Pending: {dlq.get_pending_count()}')
print(f'By type: {dlq.get_error_summary()}')
"

# Or use the dashboard
python scripts/dlq_dashboard.py --pending --limit 10

# Check slow handlers
grep "HANDLER_TIMEOUT" logs/coordination.log | tail -10
```

**Fixes**:

```python
# Increase handler timeout
import os
os.environ["RINGRIFT_HANDLER_TIMEOUT"] = "60"  # 60 seconds

# Retry DLQ events
from app.coordination.dead_letter_queue import get_dead_letter_queue
dlq = get_dead_letter_queue()
await dlq.retry_all()

# Or clear old events
dlq.cleanup(older_than_hours=24)
```

## Event Flow Verification

### Check Complete Pipeline

```bash
#!/bin/bash
# verify_feedback_loops.sh

echo "=== Checking Feedback Loop Event Flow ==="

# Check event subscriptions
python -c "
from app.coordination.event_router import get_event_bus
bus = get_event_bus()
subs = bus.list_subscribers()

required_wiring = {
    'training_completed': ['FeedbackLoopController'],
    'evaluation_completed': ['PromotionController', 'CurriculumIntegration'],
    'model_promoted': ['UnifiedDistributionDaemon'],
    'curriculum_rebalanced': ['SelfplayScheduler'],
    'quality_feedback_adjusted': ['DataPipelineOrchestrator'],
}

for event, expected in required_wiring.items():
    actual = subs.get(event, [])
    status = '✓' if len(actual) > 0 else '✗'
    print(f'{status} {event}: {len(actual)} subscriber(s)')
"

# Check daemon states
curl -s http://localhost:8770/status | python3 -c '
import sys, json
d = json.load(sys.stdin)
for daemon in ["FEEDBACK_LOOP", "CURRICULUM_INTEGRATION", "GAUNTLET_FEEDBACK", "QUALITY_MONITOR"]:
    info = d.get("daemons", {}).get(daemon, {})
    state = info.get("state", "NOT_FOUND")
    status = "✓" if state == "RUNNING" else "✗"
    print(f"{status} {daemon}: {state}")
'

echo "=== Done ==="
```

### Manual Event Injection

```python
# Test feedback loop by injecting events
from app.distributed.data_events import DataEvent, DataEventType
from app.coordination.event_router import get_event_bus

bus = get_event_bus()

# Simulate training completion
event = DataEvent(
    event_type=DataEventType.TRAINING_COMPLETED,
    payload={
        "config_key": "hex8_2p",
        "model_path": "models/test_model.pth",
        "epochs_completed": 50,
        "final_loss": 0.05,
    },
    source="test",
)
bus.publish(event)
print("Injected TRAINING_COMPLETED event")

# Check if gauntlet was triggered
import time
time.sleep(5)
# Check logs for gauntlet trigger
```

## Metrics to Monitor

| Metric                  | Location                     | Healthy Range |
| ----------------------- | ---------------------------- | ------------- |
| Feedback event rate     | `/status → event_stats`      | 1-10/hour     |
| Handler timeout rate    | `/status → handler_timeouts` | < 1%          |
| DLQ size                | `/status → dlq_size`         | < 10          |
| Curriculum update rate  | Logs                         | 1-4/day       |
| Quality adjustment rate | Logs                         | 0-2/day       |

## Environment Variables

| Variable                             | Default | Description                         |
| ------------------------------------ | ------- | ----------------------------------- |
| `RINGRIFT_FEEDBACK_ENABLED`          | true    | Enable feedback loops               |
| `RINGRIFT_HANDLER_TIMEOUT`           | 30      | Event handler timeout (seconds)     |
| `RINGRIFT_CURRICULUM_CHECK_INTERVAL` | 120     | Curriculum check interval (seconds) |
| `RINGRIFT_QUALITY_CHECK_INTERVAL`    | 300     | Quality check interval (seconds)    |
| `RINGRIFT_HYPERPARAMETER_FEEDBACK`   | true    | Enable hyperparameter feedback      |
| `RINGRIFT_DLQ_RETENTION_HOURS`       | 24      | DLQ retention period                |

## Related Runbooks

- [MODEL_PROMOTION_WORKFLOW.md](MODEL_PROMOTION_WORKFLOW.md) - Model promotion issues
- [COORDINATION_EVENT_SYSTEM.md](COORDINATION_EVENT_SYSTEM.md) - Event system reference
- [TRAINING_LOOP_STALLED.md](TRAINING_LOOP_STALLED.md) - Training issues
- [EVENT_WIRING_VERIFICATION.md](EVENT_WIRING_VERIFICATION.md) - Event subscription verification
- [DAEMON_MANAGER_OPERATIONS.md](DAEMON_MANAGER_OPERATIONS.md) - Daemon lifecycle
