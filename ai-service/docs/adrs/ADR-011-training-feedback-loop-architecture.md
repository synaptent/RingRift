# ADR-011: Training Feedback Loop Architecture

## Status

Accepted (December 2025)

## Context

The training pipeline needed automated feedback loops to:

1. Adjust training parameters based on evaluation results
2. Rebalance selfplay allocation when some configs stall
3. Detect and recover from quality regressions
4. Dynamically boost exploration when training plateaus

Previously, operators manually adjusted parameters. As the cluster scaled to 36+ nodes and 12 board configurations, manual tuning became impractical.

## Decision

We implemented a multi-layer feedback loop architecture with clear separation of concerns:

### Layer 1: Event Emission (Sources)

```
TrainingCoordinator       → TRAINING_COMPLETED, TRAINING_FAILED
EvaluationDaemon          → EVALUATION_COMPLETED, EVALUATION_FAILED
QualityMonitorDaemon      → QUALITY_SCORE_UPDATED, QUALITY_DEGRADED
ModelPerformanceWatchdog  → REGRESSION_DETECTED, REGRESSION_CLEARED
SelfplayScheduler         → SELFPLAY_COMPLETE, CURRICULUM_REBALANCED
```

### Layer 2: Feedback Controllers (Processing)

```
FeedbackLoopController    → Central orchestration, stall detection
GauntletFeedbackController → Hyperparameter adjustment post-gauntlet
CurriculumIntegration     → Curriculum weight updates
EvaluationFeedbackHandler → Real-time LR adjustments
```

### Layer 3: Effect Application (Targets)

```
SelfplayScheduler         → EXPLORATION_BOOST, CURRICULUM_REBALANCED
TrainingTriggerDaemon     → ADAPTIVE_PARAMS_CHANGED
UnifiedDistributionDaemon → MODEL_PROMOTED
```

## Key Patterns

### 1. Event-Driven Decoupling

Controllers subscribe to events rather than polling. This enables:

- Multiple subscribers per event (e.g., both FeedbackLoopController and CurriculumIntegration handle EVALUATION_COMPLETED)
- Loose coupling between components
- Easy testing via event mocking

### 2. Canonical Feedback State

`FeedbackState` dataclass hierarchy in `app/coordination/feedback_state.py`:

```python
CanonicalFeedbackState     # Base (22 fields)
├── SignalFeedbackState    # +6 fields for orchestration
└── MonitoringFeedbackState # +9 fields, +4 methods
```

Each config_key has one FeedbackState tracking:

- Quality scores, Elo ratings, parity status
- Training intensity (paused → hot_path spectrum)
- Exploration boost multiplier

### 3. Stall Detection & Recovery

`FeedbackLoopController._check_for_stalls()` runs every 10 minutes:

```
IF elo_velocity < 5 for 6+ hours:
    EMIT EXPLORATION_BOOST (temp *= 1.5)

IF quality_score < 0.3:
    EMIT QUALITY_PENALTY_APPLIED (selfplay rate *= 0.5)

IF parity_failed:
    EMIT TRAINING_BLOCKED_BY_QUALITY
```

### 4. Curriculum Rebalancing

`CurriculumIntegration` adjusts config weights based on:

- Elo velocity (faster improvement → higher weight)
- Data deficit (fewer games → higher weight)
- Quality score (lower quality → reduced weight)

## Implementation Files

| File                              | LOC   | Purpose                                 |
| --------------------------------- | ----- | --------------------------------------- |
| `feedback_loop_controller.py`     | 2,100 | Central feedback orchestration          |
| `gauntlet_feedback_controller.py` | 800   | Post-gauntlet hyperparameter adjustment |
| `curriculum_integration.py`       | 1,200 | Curriculum weight management            |
| `feedback_state.py`               | 400   | Canonical state dataclasses             |
| `selfplay_scheduler.py`           | 1,500 | Priority-based selfplay allocation      |
| `quality_monitor_daemon.py`       | 600   | Data quality tracking                   |

## Event Flow Diagram

```
TRAINING_COMPLETED
    │
    ├──→ FeedbackLoopController._trigger_evaluation()
    │         │
    │         └──→ run_gauntlet()
    │                   │
    │                   └──→ EVALUATION_COMPLETED
    │                              │
    │                              ├──→ CurriculumIntegration.handle_eval()
    │                              │         │
    │                              │         └──→ CURRICULUM_REBALANCED
    │                              │                    │
    │                              │                    └──→ SelfplayScheduler
    │                              │
    │                              └──→ GauntletFeedbackController.analyze()
    │                                        │
    │                                        └──→ HYPERPARAMETER_UPDATED
    │                                                   │
    │                                                   └──→ EvaluationFeedbackHandler
    │
    └──→ ModelPerformanceWatchdog.track()
              │
              └──→ REGRESSION_DETECTED (if win_rate drops)
                         │
                         └──→ ModelLifecycleCoordinator (rollback)
```

## Consequences

### Positive

- Fully automated training lifecycle with no manual intervention
- Self-healing: stalled configs get exploration boost automatically
- Quality gates prevent training on bad data
- Traceable: all decisions logged with event IDs

### Negative

- Complex event dependencies require careful wiring (see `coordination_bootstrap.py`)
- Debugging requires understanding event flow
- State is distributed across multiple FeedbackState instances

## Alternatives Considered

1. **Centralized Controller**: Single monolithic controller
   - Rejected: Would grow too large, harder to test

2. **Polling-Based**: Controllers poll for state changes
   - Rejected: Wasteful, misses transient events

3. **External Scheduler** (Airflow, Prefect)
   - Rejected: Overkill for internal pipeline, added operational burden

## Testing Strategy

- Unit tests: Each controller tested with mocked events
- Integration tests: `test_daemon_startup_order.py` verifies wiring
- Smoke tests: `test_coordination_smoke.py` checks event subscriptions

## Related ADRs

- ADR-001: Event-Driven Architecture (foundation)
- ADR-002: Daemon Lifecycle Management (controller lifecycle)
- ADR-009: Daemon Registry Pattern (controller registration)
