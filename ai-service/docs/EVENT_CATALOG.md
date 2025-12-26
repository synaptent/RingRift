# Event Catalog

Complete reference of all event types in the RingRift AI training infrastructure.

**Last Updated:** December 2025
**Total Event Types:** 159
**Source of Truth:** `app/events/types.py`

## Quick Reference

| Category     | Count | Purpose                   |
| ------------ | ----- | ------------------------- |
| Data         | 9     | Data sync, freshness      |
| Training     | 10    | Training lifecycle        |
| Evaluation   | 8     | Model evaluation          |
| Promotion    | 8     | Model promotion           |
| Curriculum   | 3     | Curriculum management     |
| Selfplay     | 3     | Selfplay operations       |
| Optimization | 12    | CMA-ES, NAS, PBT          |
| Quality      | 9     | Data quality              |
| Regression   | 6     | Regression detection      |
| Cluster      | 12    | P2P/cluster ops           |
| Work         | 8     | Work queue                |
| Stage        | 20    | Pipeline stages           |
| System       | 40+   | Health, daemons, recovery |

## Usage

```python
from app.events import RingRiftEventType, EventCategory

# Use unified event types
if event.event_type == RingRiftEventType.MODEL_PROMOTED:
    handle_promotion(event)

# Check event category
category = EventCategory.from_event(event.event_type)
if category == EventCategory.TRAINING:
    log_training_event(event)

# Backwards compatibility
from app.events import DataEventType  # Alias for RingRiftEventType
```

---

## Event Categories

### DATA Events

Events related to data collection, sync, and freshness.

| Event                 | Value            | Description                  |
| --------------------- | ---------------- | ---------------------------- |
| `NEW_GAMES_AVAILABLE` | `new_games`      | New games ready for training |
| `DATA_SYNC_STARTED`   | `sync_started`   | Sync operation started       |
| `DATA_SYNC_COMPLETED` | `sync_completed` | Sync operation completed     |
| `DATA_SYNC_FAILED`    | `sync_failed`    | Sync operation failed        |
| `GAME_SYNCED`         | `game_synced`    | Individual game synced       |
| `DATA_STALE`          | `data_stale`     | Data freshness warning       |
| `DATA_FRESH`          | `data_fresh`     | Data freshness OK            |
| `SYNC_TRIGGERED`      | `sync_triggered` | Sync manually triggered      |
| `SYNC_STALLED`        | `sync_stalled`   | Sync appears stuck           |

**Payload Example:**

```python
{
    "event_type": "sync_completed",
    "source": "auto_sync_daemon",
    "payload": {
        "host": "lambda-gh200-a",
        "games_synced": 150,
        "bytes_transferred": 52428800,
        "duration_seconds": 12.5
    }
}
```

---

### TRAINING Events

Events related to training lifecycle.

| Event                         | Value                         | Description                    |
| ----------------------------- | ----------------------------- | ------------------------------ |
| `TRAINING_THRESHOLD_REACHED`  | `training_threshold`          | Games threshold met            |
| `TRAINING_STARTED`            | `training_started`            | Training job started           |
| `TRAINING_PROGRESS`           | `training_progress`           | Training progress update       |
| `TRAINING_COMPLETED`          | `training_completed`          | Training finished successfully |
| `TRAINING_FAILED`             | `training_failed`             | Training failed                |
| `TRAINING_LOSS_ANOMALY`       | `training_loss_anomaly`       | Loss spike detected            |
| `TRAINING_LOSS_TREND`         | `training_loss_trend`         | Loss trend analysis            |
| `TRAINING_EARLY_STOPPED`      | `training_early_stopped`      | Early stopping triggered       |
| `TRAINING_ROLLBACK_NEEDED`    | `training_rollback_needed`    | Rollback recommended           |
| `TRAINING_ROLLBACK_COMPLETED` | `training_rollback_completed` | Rollback finished              |

**Payload Example:**

```python
{
    "event_type": "training_completed",
    "payload": {
        "config_key": "hex8_2p",
        "model_path": "models/hex8_2p/checkpoint_epoch_50.pth",
        "epochs": 50,
        "final_loss": 0.423,
        "policy_accuracy": 0.762,
        "value_accuracy": 0.891,
        "duration_seconds": 3600
    }
}
```

---

### EVALUATION Events

Events related to model evaluation.

| Event                     | Value                     | Description                  |
| ------------------------- | ------------------------- | ---------------------------- |
| `EVALUATION_STARTED`      | `evaluation_started`      | Evaluation begun             |
| `EVALUATION_PROGRESS`     | `evaluation_progress`     | Evaluation progress          |
| `EVALUATION_COMPLETED`    | `evaluation_completed`    | Evaluation finished          |
| `EVALUATION_FAILED`       | `evaluation_failed`       | Evaluation failed            |
| `ELO_UPDATED`             | `elo_updated`             | ELO rating changed           |
| `ELO_SIGNIFICANT_CHANGE`  | `elo_significant_change`  | Large ELO change (>20)       |
| `ELO_VELOCITY_CHANGED`    | `elo_velocity_changed`    | ELO improvement rate changed |
| `ADAPTIVE_PARAMS_CHANGED` | `adaptive_params_changed` | Adaptive parameters updated  |

**Payload Example:**

```python
{
    "event_type": "evaluation_completed",
    "payload": {
        "config_key": "hex8_2p",
        "model_path": "models/hex8_2p/best.pth",
        "games_played": 100,
        "win_rate_vs_heuristic": 0.72,
        "win_rate_vs_random": 0.95,
        "elo_rating": 1720,
        "elo_delta": 45
    }
}
```

---

### PROMOTION Events

Events related to model promotion.

| Event                   | Value                   | Description                  |
| ----------------------- | ----------------------- | ---------------------------- |
| `PROMOTION_CANDIDATE`   | `promotion_candidate`   | Model eligible for promotion |
| `PROMOTION_STARTED`     | `promotion_started`     | Promotion process started    |
| `MODEL_PROMOTED`        | `model_promoted`        | Model promoted to production |
| `PROMOTION_FAILED`      | `promotion_failed`      | Promotion failed             |
| `PROMOTION_REJECTED`    | `promotion_rejected`    | Model didn't meet criteria   |
| `PROMOTION_ROLLED_BACK` | `promotion_rolled_back` | Promotion reverted           |
| `TIER_PROMOTION`        | `tier_promotion`        | ELO tier promotion           |
| `CROSSBOARD_PROMOTION`  | `crossboard_promotion`  | Cross-board transfer         |

**Payload Example:**

```python
{
    "event_type": "model_promoted",
    "payload": {
        "config_key": "hex8_2p",
        "model_path": "models/canonical_hex8_2p.pth",
        "previous_elo": 1650,
        "new_elo": 1720,
        "win_rate_vs_previous": 0.58
    }
}
```

---

### CURRICULUM Events

Events related to curriculum management.

| Event                   | Value                   | Description               |
| ----------------------- | ----------------------- | ------------------------- |
| `CURRICULUM_REBALANCED` | `curriculum_rebalanced` | Weights redistributed     |
| `CURRICULUM_ADVANCED`   | `curriculum_advanced`   | Stage advanced            |
| `WEIGHT_UPDATED`        | `weight_updated`        | Individual weight changed |

**Payload Example:**

```python
{
    "event_type": "curriculum_rebalanced",
    "payload": {
        "weights": {
            "hex8_2p": 1.5,
            "square8_2p": 1.0,
            "hex8_4p": 0.8
        },
        "reason": "elo_velocity_adjustment"
    }
}
```

---

### SELFPLAY Events

Events related to selfplay operations.

| Event                     | Value                     | Description             |
| ------------------------- | ------------------------- | ----------------------- |
| `SELFPLAY_TARGET_UPDATED` | `selfplay_target_updated` | Target games changed    |
| `SELFPLAY_RATE_CHANGED`   | `selfplay_rate_changed`   | Games/hour rate changed |
| `IDLE_RESOURCE_DETECTED`  | `idle_resource_detected`  | Idle GPU found          |

---

### QUALITY Events

Events related to data quality.

| Event                          | Value                          | Description            |
| ------------------------------ | ------------------------------ | ---------------------- |
| `DATA_QUALITY_ALERT`           | `data_quality_alert`           | Quality issue detected |
| `QUALITY_CHECK_FAILED`         | `quality_check_failed`         | Quality check failed   |
| `QUALITY_SCORE_UPDATED`        | `quality_score_updated`        | Score recalculated     |
| `QUALITY_DISTRIBUTION_CHANGED` | `quality_distribution_changed` | Distribution shift     |
| `HIGH_QUALITY_DATA_AVAILABLE`  | `high_quality_data_available`  | HQ data ready          |
| `QUALITY_DEGRADED`             | `quality_degraded`             | Quality declining      |
| `LOW_QUALITY_DATA_WARNING`     | `low_quality_data_warning`     | Quality warning        |
| `TRAINING_BLOCKED_BY_QUALITY`  | `training_blocked_by_quality`  | Training paused        |
| `QUALITY_PENALTY_APPLIED`      | `quality_penalty_applied`      | Penalty applied        |

---

### REGRESSION Events

Events related to regression detection.

| Event                 | Value                 | Description         |
| --------------------- | --------------------- | ------------------- |
| `REGRESSION_DETECTED` | `regression_detected` | Regression found    |
| `REGRESSION_MINOR`    | `regression_minor`    | Small regression    |
| `REGRESSION_MODERATE` | `regression_moderate` | Medium regression   |
| `REGRESSION_SEVERE`   | `regression_severe`   | Large regression    |
| `REGRESSION_CRITICAL` | `regression_critical` | Critical regression |
| `REGRESSION_CLEARED`  | `regression_cleared`  | Regression resolved |

---

### CLUSTER Events

Events related to P2P and cluster operations.

| Event                      | Value                      | Description          |
| -------------------------- | -------------------------- | -------------------- |
| `P2P_MODEL_SYNCED`         | `p2p_model_synced`         | Model synced via P2P |
| `P2P_CLUSTER_HEALTHY`      | `p2p_cluster_healthy`      | Cluster health OK    |
| `P2P_CLUSTER_UNHEALTHY`    | `p2p_cluster_unhealthy`    | Cluster issues       |
| `P2P_NODES_DEAD`           | `p2p_nodes_dead`           | Nodes unreachable    |
| `P2P_SELFPLAY_SCALED`      | `p2p_selfplay_scaled`      | Selfplay scaled      |
| `CLUSTER_STATUS_CHANGED`   | `cluster_status_changed`   | Status change        |
| `CLUSTER_CAPACITY_CHANGED` | `cluster_capacity_changed` | Capacity change      |
| `NODE_UNHEALTHY`           | `node_unhealthy`           | Node health issue    |
| `NODE_RECOVERED`           | `node_recovered`           | Node back online     |
| `NODE_ACTIVATED`           | `node_activated`           | Node started         |
| `NODE_CAPACITY_UPDATED`    | `node_capacity_updated`    | Capacity update      |
| `NODE_OVERLOADED`          | `node_overloaded`          | Node overloaded      |

---

### WORK Queue Events

Events related to work queue operations.

| Event            | Value            | Description            |
| ---------------- | ---------------- | ---------------------- |
| `WORK_QUEUED`    | `work_queued`    | Work item added        |
| `WORK_CLAIMED`   | `work_claimed`   | Work claimed by worker |
| `WORK_STARTED`   | `work_started`   | Work execution started |
| `WORK_COMPLETED` | `work_completed` | Work finished          |
| `WORK_FAILED`    | `work_failed`    | Work failed            |
| `WORK_RETRY`     | `work_retry`     | Work being retried     |
| `WORK_TIMEOUT`   | `work_timeout`   | Work timed out         |
| `WORK_CANCELLED` | `work_cancelled` | Work cancelled         |

---

### STAGE Events

Pipeline stage completion events.

| Event                              | Value                        | Description         |
| ---------------------------------- | ---------------------------- | ------------------- |
| `STAGE_SELFPLAY_COMPLETE`          | `selfplay_complete`          | Selfplay stage done |
| `STAGE_SYNC_COMPLETE`              | `sync_complete`              | Sync stage done     |
| `STAGE_PARITY_VALIDATION_COMPLETE` | `parity_validation_complete` | Parity check done   |
| `STAGE_NPZ_EXPORT_COMPLETE`        | `npz_export_complete`        | Export done         |
| `STAGE_TRAINING_COMPLETE`          | `stage_training_complete`    | Training stage done |
| `STAGE_EVALUATION_COMPLETE`        | `stage_evaluation_complete`  | Eval stage done     |
| `STAGE_PROMOTION_COMPLETE`         | `stage_promotion_complete`   | Promotion done      |
| `STAGE_ITERATION_COMPLETE`         | `iteration_complete`         | Full iteration done |

---

## Cross-Process Events

Events that should propagate across process boundaries:

```python
CROSS_PROCESS_EVENT_TYPES = {
    # Success events
    MODEL_PROMOTED,
    TIER_PROMOTION,
    TRAINING_STARTED,
    TRAINING_COMPLETED,
    EVALUATION_COMPLETED,
    CURRICULUM_REBALANCED,
    ELO_SIGNIFICANT_CHANGE,
    DATA_SYNC_COMPLETED,

    # Failure events
    TRAINING_FAILED,
    EVALUATION_FAILED,
    PROMOTION_FAILED,
    DATA_SYNC_FAILED,

    # System events
    HOST_ONLINE,
    HOST_OFFLINE,
    DAEMON_STARTED,
    DAEMON_STOPPED,

    # Regression events
    REGRESSION_DETECTED,
    REGRESSION_SEVERE,
    REGRESSION_CRITICAL,
}
```

---

## Event Flow Examples

### Training Pipeline

```
NEW_GAMES_AVAILABLE
    → TRAINING_THRESHOLD_REACHED
    → TRAINING_STARTED
    → TRAINING_PROGRESS (multiple)
    → TRAINING_COMPLETED
    → EVALUATION_STARTED
    → EVALUATION_COMPLETED
    → PROMOTION_CANDIDATE
    → MODEL_PROMOTED
    → CURRICULUM_REBALANCED
```

### Regression Detection

```
EVALUATION_COMPLETED (win_rate < threshold)
    → ELO_UPDATED (negative delta)
    → REGRESSION_DETECTED
    → TRAINING_ROLLBACK_NEEDED
    → TRAINING_ROLLBACK_COMPLETED
    → REGRESSION_CLEARED
```

### Data Sync Flow

```
IDLE_RESOURCE_DETECTED
    → SELFPLAY_TARGET_UPDATED
    → NEW_GAMES_AVAILABLE
    → DATA_SYNC_STARTED
    → GAME_SYNCED (multiple)
    → DATA_SYNC_COMPLETED
    → DATA_FRESH
```

---

## Publishing Events

```python
from app.coordination.event_router import get_router, publish

router = get_router()

# Publish an event
await router.publish(
    event_type=RingRiftEventType.TRAINING_COMPLETED,
    payload={
        "config_key": "hex8_2p",
        "model_path": "models/hex8_2p/best.pth",
        "epochs": 50,
    },
    source="training_daemon",
    cross_process=True,  # Propagate across processes
)
```

---

## Subscribing to Events

```python
from app.coordination.event_router import get_router, subscribe

router = get_router()

# Subscribe to an event
async def on_model_promoted(event):
    print(f"Model promoted: {event.payload}")

unsubscribe = subscribe(
    RingRiftEventType.MODEL_PROMOTED,
    on_model_promoted,
)

# Later: unsubscribe()
```
