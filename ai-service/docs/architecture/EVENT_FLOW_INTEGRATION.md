# Event Flow Integration Map

**Last Updated**: December 29, 2025

This document describes the complete event flow paths in the RingRift training pipeline,
from selfplay generation through model promotion.

## Overview

The training pipeline is fully event-driven with 6 major flow paths:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RingRift Training Pipeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   [Selfplay] ──NEW_GAMES_AVAILABLE──► [Data Pipeline] ──TRAINING_THRESHOLD──►│
│                                              │                               │
│                                              ▼                               │
│   [Feedback] ◄──EVALUATION_COMPLETED── [Evaluation] ◄──TRAINING_COMPLETED── │
│       │                                      │                               │
│       ▼                                      ▼                               │
│  [Curriculum] ◄──MODEL_PROMOTED────── [Promotion] ◄───────────────────────  │
│       │                                      │                               │
│       └────────►[Selfplay Scheduler]◄────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Selfplay → Training Flow

### Event Chain

```
SelfplayRunner
    │
    ├── SELFPLAY_COMPLETE
    │       ├──► DataConsolidationDaemon._on_selfplay_complete()
    │       ├──► UnifiedFeedback._on_selfplay_complete()
    │       ├──► UnifiedIdleShutdownDaemon._on_selfplay_completed()
    │       └──► CoordinationBootstrap.on_selfplay_complete_for_sync()
    │
    └── NEW_GAMES_AVAILABLE
            ├──► DataPipelineOrchestrator._on_new_games()
            │       └── Checks if game_count >= training_threshold
            │           └── Emits: TRAINING_THRESHOLD_REACHED
            ├──► TrainingCoordinator._on_new_games_available()
            ├──► UnifiedReplicationDaemon._on_new_games()
            └──► UnifiedQueuePopulator._on_new_games()
```

### Required Payload

```python
{
    "config_key": "hex8_2p",  # Board config identifier
    "game_count": 150,        # Games generated this batch
    "total_games": 5000,      # Total games for config
    "db_path": "data/games/canonical_hex8_2p.db",
    "timestamp": 1735500000.0
}
```

---

## 2. Training → Evaluation Flow

### Event Chain

```
TrainingCoordinator
    │
    ├── TRAINING_STARTED
    │       ├──► UnifiedIdleShutdownDaemon._on_training_started()
    │       └──► UnifiedFeedback._on_training_started() (implicit)
    │
    ├── TRAINING_PROGRESS (periodic)
    │       └──► UnifiedDistributionDaemon._on_training_progress_for_prefetch()
    │               └── Prefetches model for faster distribution
    │
    └── TRAINING_COMPLETED
            ├──► FeedbackLoopController._trigger_evaluation()
            │       └── Spawns evaluation job
            │           └── Emits: EVALUATION_STARTED
            ├──► UnifiedQueuePopulator._on_training_completed()
            ├──► UnifiedFeedback._on_training_complete()
            └──► DataPipelineOrchestrator._on_training_completed()
```

### Required Payload

```python
{
    "config_key": "hex8_2p",
    "model_path": "models/canonical_hex8_2p.pth",
    "board_type": "hex8",
    "num_players": 2,
    "epochs_trained": 50,
    "final_loss": 0.234,
    "val_loss": 0.256,
    "elo_estimate": 1450,
    "timestamp": 1735500000.0
}
```

---

## 3. Evaluation → Promotion Flow

### Event Chain

```
EvaluationDaemon / GameGauntlet
    │
    ├── EVALUATION_STARTED
    │       └──► CoordinationBootstrap.on_evaluation_started()
    │
    ├── EVALUATION_COMPLETED
    │       ├──► AutoPromotionDaemon._on_evaluation_completed()
    │       │       └── Checks if elo_delta >= threshold
    │       │           └── Emits: MODEL_PROMOTED or logs skip reason
    │       ├──► CurriculumIntegration._on_evaluation_completed()
    │       ├──► UnifiedFeedback._on_evaluation_complete()
    │       └──► FeedbackLoopController._on_evaluation_complete()
    │
    └── (if backpressure)
        └── EVALUATION_BACKPRESSURE
                ├──► TrainingCoordinator._on_evaluation_backpressure()
                │       └── Pauses new training jobs
                └──► DataPipelineOrchestrator._on_backpressure()
                        └── Pauses export triggering
```

### Required Payload

```python
{
    "config_key": "hex8_2p",
    "model_path": "models/candidate_hex8_2p.pth",
    "elo_delta": 45,           # Elo improvement
    "win_rate_vs_random": 0.95,
    "win_rate_vs_heuristic": 0.65,
    "passed_gauntlet": True,
    "games_played": 100,
    "timestamp": 1735500000.0
}
```

---

## 4. Promotion → Distribution Flow

### Event Chain

```
AutoPromotionDaemon / PromotionController
    │
    └── MODEL_PROMOTED
            ├──► UnifiedDistributionDaemon._on_model_promoted()
            │       ├── Distributes model to all nodes
            │       │       └── Uses: BitTorrent > HTTP > rsync
            │       └── Emits: MODEL_DISTRIBUTION_COMPLETE
            ├──► CurriculumIntegration._on_model_promoted()
            │       └── Updates curriculum weights
            ├──► CacheCoordinationOrchestrator._on_model_promoted()
            │       └── Invalidates old model caches
            └──► UnifiedFeedback._on_promotion_complete()
                    └── Closes feedback loop
```

### Required Payload

```python
{
    "config_key": "hex8_2p",
    "model_path": "models/canonical_hex8_2p.pth",
    "previous_elo": 1400,
    "new_elo": 1450,
    "promotion_reason": "gauntlet_passed",
    "timestamp": 1735500000.0
}
```

---

## 5. Regression → Recovery Flow

### Event Chain

```
RegressionDetector
    │
    ├── REGRESSION_DETECTED (minor regression)
    │       ├──► TrainingCoordinator._on_regression_detected()
    │       │       └── Reduces learning rate
    │       ├──► UnifiedFeedback._on_regression_detected()
    │       │       └── Adjusts training parameters
    │       └──► UnifiedHealthManager._on_regression_detected()
    │               └── Updates health score
    │
    └── REGRESSION_CRITICAL (severe regression)
            ├──► TrainingCoordinator._on_regression_critical()
            │       └── Pauses training immediately
            ├──► CurriculumIntegration._on_regression_critical()
            │       └── Emits: CURRICULUM_EMERGENCY_UPDATE
            ├──► DaemonManager._on_regression_critical()
            │       └── May restart affected daemons
            └──► UnifiedHealthManager._on_regression_critical()
                    └── Critical health alert

(If rollback needed)
    │
    └── TRAINING_ROLLBACK_NEEDED
            └──► TrainingCoordinator._handle_rollback()
                    └── Reverts to previous checkpoint
                        └── Emits: PROMOTION_ROLLED_BACK
```

---

## 6. Cluster Health Flow

### Event Chain

```
P2P Orchestrator / NodeMonitor
    │
    ├── HOST_OFFLINE
    │       ├──► UnifiedHealthManager._on_host_offline()
    │       │       └── Updates cluster health score
    │       ├──► ClusterWatchdogDaemon._on_host_offline()
    │       │       └── Updates cluster view
    │       ├──► UnifiedReplicationDaemon._on_host_offline()
    │       │       └── Adjusts replication targets
    │       └──► DaemonEventHandlers._on_host_offline()
    │               └── May trigger failover
    │
    ├── P2P_CLUSTER_UNHEALTHY (majority offline)
    │       ├──► TrainingCoordinator._on_cluster_unhealthy()
    │       │       └── Pauses training
    │       ├──► AutoScaler._on_cluster_unhealthy()
    │       │       └── Pauses scaling
    │       └──► SelfplayScheduler._on_cluster_unhealthy()
    │               └── Reduces selfplay rate
    │
    └── LEADER_ELECTED
            └──► DaemonEventHandlers._on_leader_elected()
                    └── Adjusts daemon roles for new leader
```

---

## Integration Test Scenarios

### Scenario 1: Full Training Cycle

```python
async def test_full_training_cycle():
    """Test complete selfplay → training → eval → promotion."""
    # 1. Emit selfplay completion
    bus.emit(DataEventType.SELFPLAY_COMPLETE, {...})

    # 2. Wait for new games event
    await wait_for_event(DataEventType.NEW_GAMES_AVAILABLE)

    # 3. Wait for training threshold
    await wait_for_event(DataEventType.TRAINING_THRESHOLD_REACHED)

    # 4. Wait for training to complete
    await wait_for_event(DataEventType.TRAINING_COMPLETED)

    # 5. Wait for evaluation
    await wait_for_event(DataEventType.EVALUATION_COMPLETED)

    # 6. Check if model was promoted
    promoted = await wait_for_event(DataEventType.MODEL_PROMOTED, timeout=60)
    assert promoted["elo_delta"] > 0
```

### Scenario 2: Regression Recovery

```python
async def test_regression_recovery():
    """Test regression detection and recovery."""
    # 1. Emit critical regression
    bus.emit(DataEventType.REGRESSION_CRITICAL, {
        "config_key": "hex8_2p",
        "elo_drop": 150,
    })

    # 2. Verify training paused
    assert training_coordinator.is_paused()

    # 3. Wait for rollback
    await wait_for_event(DataEventType.PROMOTION_ROLLED_BACK)

    # 4. Verify training resumes
    await asyncio.sleep(5)
    assert not training_coordinator.is_paused()
```

### Scenario 3: Cluster Health Degradation

```python
async def test_cluster_health_degradation():
    """Test cluster health event handling."""
    # 1. Emit host offline events
    for i in range(3):
        bus.emit(DataEventType.HOST_OFFLINE, {"node_id": f"node-{i}"})

    # 2. Wait for cluster unhealthy
    await wait_for_event(DataEventType.P2P_CLUSTER_UNHEALTHY)

    # 3. Verify training paused
    assert training_coordinator.is_paused()

    # 4. Emit hosts back online
    for i in range(3):
        bus.emit(DataEventType.HOST_ONLINE, {"node_id": f"node-{i}"})

    # 5. Wait for cluster healthy
    await wait_for_event(DataEventType.P2P_CLUSTER_HEALTHY)

    # 6. Verify training resumed
    assert not training_coordinator.is_paused()
```

---

## Event Ordering Guarantees

### Strict Ordering (Within Same Flow)

- `TRAINING_STARTED` always before `TRAINING_COMPLETED`
- `EVALUATION_STARTED` always before `EVALUATION_COMPLETED`
- `MODEL_PROMOTED` always before `MODEL_DISTRIBUTION_COMPLETE`

### No Ordering (Cross-Flow)

- Multiple `SELFPLAY_COMPLETE` events may arrive concurrently
- `TRAINING_COMPLETED` from different configs is independent
- Health events are asynchronous

### Idempotency

All handlers should be idempotent. Duplicate events may occur due to:

- Network retries
- Event replay from DLQ
- Multiple emitters for same event

---

## Error Handling

### Event Handler Failures

```python
# In event_router.py
async def _dispatch_with_retry(event, handler):
    for attempt in range(3):
        try:
            await handler(event)
            return
        except Exception as e:
            if attempt == 2:
                await self.dlq.enqueue(event, error=str(e))
            else:
                await asyncio.sleep(2 ** attempt)
```

### Dead Letter Queue (DLQ)

Failed events are stored in DLQ with:

- Original event payload
- Error message
- Retry count
- Timestamp

Replay failed events:

```python
from app.coordination.dead_letter_queue import replay_dlq_events
await replay_dlq_events(max_age_hours=24)
```

---

## See Also

- `EVENT_SUBSCRIPTION_MATRIX.md` - Complete event list
- `DAEMON_LIFECYCLE.md` - Daemon state machines
- `app/coordination/event_router.py` - Event bus implementation
- `app/distributed/data_events/event_types.py` - Event type definitions
