# Feedback Loop Degradation Runbook

This runbook covers diagnosis and resolution of feedback loop degradation - when the training feedback system stops effectively adjusting training parameters.

**Created**: December 28, 2025
**Version**: 1.0
**Severity**: Medium

---

## Overview

The feedback loop system adjusts training parameters based on:

- Evaluation results (gauntlet win rates)
- Elo velocity (rate of Elo improvement)
- Data quality scores
- Curriculum progression

Degradation occurs when these signals stop influencing training behavior.

---

## Detection Methods

### Method 1: Check Feedback Controller Health

```python
from app.coordination.feedback_loop_controller import get_feedback_controller

controller = get_feedback_controller()
health = controller.health_check()

print(f"Healthy: {health.is_healthy}")
print(f"Active configs: {health.details.get('active_configs', [])}")
print(f"Last feedback: {health.details.get('last_feedback_time')}")
print(f"Events received: {health.details.get('events_received', 0)}")
```

### Method 2: Check Event Flow

```bash
# Check feedback-related events
grep -E "(EVALUATION_COMPLETED|ELO_VELOCITY|QUALITY_SCORE)" logs/coordination.log | tail -20

# Check if controller is subscribed
grep "FeedbackLoopController.*subscri" logs/coordination.log | tail -5
```

### Method 3: Monitor Elo Progression

```python
from app.coordination.elo_sync_manager import get_elo_sync_manager

mgr = get_elo_sync_manager()
history = mgr.get_elo_history("hex8_2p", days=7)

# Check for Elo stagnation
if len(history) > 2:
    start_elo = history[0]["elo"]
    end_elo = history[-1]["elo"]
    velocity = (end_elo - start_elo) / 7  # Elo/day
    print(f"Elo velocity: {velocity:.1f}/day")
    if velocity < 5:
        print("WARNING: Low Elo velocity - feedback may be degraded")
```

---

## Degradation Patterns

### Pattern 1: No Events Received

**Symptom**: Controller receives zero events.

**Diagnosis**:

```python
controller = get_feedback_controller()
print(f"Events received: {controller.stats.events_received}")
print(f"Subscribed: {controller._is_subscribed}")
```

**Fix**:

```python
# Re-subscribe to events
from app.coordination.coordination_bootstrap import wire_feedback_loop

await wire_feedback_loop()
```

---

### Pattern 2: Events Received But No Action

**Symptom**: Events arrive but parameters don't change.

**Diagnosis**:

```python
controller = get_feedback_controller()
for config, state in controller.get_all_states().items():
    print(f"\n{config}:")
    print(f"  Exploration boost: {state.exploration_boost}")
    print(f"  Training intensity: {state.training_intensity}")
    print(f"  Last update: {state.last_update_time}")
```

**Common Causes**:

- Thresholds too strict
- Already at optimal settings
- Signal below noise threshold

**Fix**:

```python
# Lower feedback thresholds
import os
os.environ["RINGRIFT_FEEDBACK_ELO_VELOCITY_MIN"] = "5"
os.environ["RINGRIFT_FEEDBACK_QUALITY_THRESHOLD"] = "0.5"

# Force a feedback cycle
await controller.force_feedback_update("hex8_2p")
```

---

### Pattern 3: Plateau Detection Failure

**Symptom**: Training plateaus but curriculum doesn't advance.

**Diagnosis**:

```bash
# Check plateau detection
grep "plateau\|PLATEAU" logs/coordination.log | tail -10
```

**Fix**:

```python
from app.coordination.feedback_loop_controller import get_feedback_controller

controller = get_feedback_controller()

# Manually detect plateau
state = controller.get_state("hex8_2p")
if state.elo_velocity < 5 and state.epochs_since_improvement > 10:
    await controller.trigger_plateau_response("hex8_2p")
```

---

### Pattern 4: Gauntlet Not Triggering

**Symptom**: Training completes but no evaluation runs.

**Diagnosis**:

```python
from app.coordination.evaluation_daemon import get_evaluation_daemon

daemon = get_evaluation_daemon()
print(f"Queue depth: {daemon.get_queue_depth()}")
print(f"Last evaluation: {daemon.get_last_evaluation_time()}")
```

**Fix**:

```python
# Manually trigger gauntlet
from app.coordination.feedback_loop_controller import get_feedback_controller

controller = get_feedback_controller()
await controller.run_gauntlet("hex8_2p")
```

---

## Feedback Loop Components

### Component Health Check

```python
from app.coordination.feedback_loop_controller import get_feedback_controller
from app.coordination.gauntlet_feedback_controller import get_gauntlet_controller
from app.coordination.curriculum_feedback import get_curriculum_integration

# Check all feedback components
components = [
    ("FeedbackLoopController", get_feedback_controller()),
    ("GauntletFeedbackController", get_gauntlet_controller()),
    ("CurriculumIntegration", get_curriculum_integration()),
]

for name, comp in components:
    health = comp.health_check()
    print(f"{name}: {'OK' if health.is_healthy else 'DEGRADED'}")
```

### Event Wiring Verification

```python
from app.coordination.event_router import get_router

router = get_router()

# Check feedback-related subscriptions
feedback_events = [
    "EVALUATION_COMPLETED",
    "TRAINING_COMPLETED",
    "ELO_SIGNIFICANT_CHANGE",
    "QUALITY_SCORE_UPDATED",
]

for event in feedback_events:
    subs = router.get_subscribers(event)
    print(f"{event}: {len(subs)} subscribers")
    for sub in subs:
        print(f"  - {sub.__class__.__name__}")
```

---

## Recovery Procedures

### Option 1: Restart Feedback Controller

```python
from app.coordination.daemon_manager import get_daemon_manager
from app.coordination.daemon_types import DaemonType

dm = get_daemon_manager()

await dm.restart_daemon(DaemonType.FEEDBACK_LOOP)
await dm.restart_daemon(DaemonType.CURRICULUM_INTEGRATION)
```

### Option 2: Re-wire Event Subscriptions

```python
from app.coordination.coordination_bootstrap import (
    wire_feedback_loop,
    wire_curriculum_integration,
)

# Re-wire all feedback components
await wire_feedback_loop()
await wire_curriculum_integration()
```

### Option 3: Manual Feedback Injection

```python
from app.coordination.feedback_loop_controller import get_feedback_controller
from app.coordination.feedback_state import SignalFeedbackState

controller = get_feedback_controller()

# Manually update feedback state
state = SignalFeedbackState(config_key="hex8_2p")
state.elo_velocity = 10.0
state.quality_score = 0.85
state.exploration_boost = 1.2
state.training_intensity = "accelerated"

controller.update_state("hex8_2p", state)
```

### Option 4: Reset Feedback State

```python
from app.coordination.feedback_loop_controller import get_feedback_controller

controller = get_feedback_controller()

# Reset to defaults
controller.reset_state("hex8_2p")

# Or reset all
controller.reset_all_states()
```

---

## Monitoring Feedback Effectiveness

### Key Metrics

| Metric               | Healthy Range | Source                 |
| -------------------- | ------------- | ---------------------- |
| elo_velocity         | > 5 Elo/day   | EloSyncManager         |
| feedback_events_hour | > 10          | FeedbackLoopController |
| exploration_boost    | 0.8 - 1.5     | FeedbackState          |
| training_intensity   | varies        | FeedbackState          |

### Dashboard Query

```python
from app.coordination.feedback_loop_controller import get_feedback_controller

controller = get_feedback_controller()

for config, state in controller.get_all_states().items():
    print(f"\n=== {config} ===")
    print(f"Elo velocity: {state.elo_velocity:.1f}/day")
    print(f"Quality score: {state.quality_score:.2f}")
    print(f"Training intensity: {state.training_intensity}")
    print(f"Exploration boost: {state.exploration_boost:.2f}")
    print(f"Curriculum tier: {state.curriculum_tier}")
```

---

## Prevention

### 1. Event Subscription Monitoring

```python
# Add to health check
def verify_feedback_subscriptions():
    from app.coordination.event_router import get_router

    required = ["EVALUATION_COMPLETED", "TRAINING_COMPLETED"]
    router = get_router()

    for event in required:
        if not router.get_subscribers(event):
            logger.warning(f"No subscribers for {event} - feedback may degrade")
```

### 2. Periodic Feedback Validation

```bash
# Add to cron or daemon
*/30 * * * * python -c "
from app.coordination.feedback_loop_controller import get_feedback_controller
controller = get_feedback_controller()
health = controller.health_check()
if not health.is_healthy:
    print(f'ALERT: Feedback loop degraded: {health.message}')
"
```

### 3. Elo Velocity Alerting

```python
# In monitoring daemon
async def check_elo_stagnation():
    from app.coordination.elo_sync_manager import get_elo_sync_manager

    mgr = get_elo_sync_manager()
    for config in ["hex8_2p", "square8_2p", "hexagonal_2p"]:
        velocity = mgr.get_elo_velocity(config, days=3)
        if velocity < 3:
            emit_alert(f"{config}: Elo stagnation detected (velocity={velocity})")
```

---

## Related Documentation

- [FEEDBACK_LOOP_TROUBLESHOOTING.md](FEEDBACK_LOOP_TROUBLESHOOTING.md) - General feedback issues
- [TRAINING_LOOP_STALLED.md](TRAINING_LOOP_STALLED.md) - Training issues
- [MODEL_PROMOTION_WORKFLOW.md](MODEL_PROMOTION_WORKFLOW.md) - Promotion flow
