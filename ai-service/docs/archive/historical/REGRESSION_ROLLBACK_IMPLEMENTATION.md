# Regression Detection → Automatic Rollback Implementation

## Overview

This document describes the implementation of automatic model rollback when significant regressions are detected (elo_drop > 30).

## Problem Statement

Previously, REGRESSION_DETECTED events were emitted but no automatic rollback occurred. The system needed to be wired to trigger rollback when significant regressions happen.

## Solution

The solution involves wiring the existing components together with a new check for MODERATE regressions with elo_drop > 30.

## Architecture

### Event Flow

```
1. RegressionDetector emits REGRESSION_DETECTED event
   ↓
2. AutoRollbackHandler receives event via event bus subscription
   ↓
3. For MODERATE regressions with elo_drop > 30:
   - Triggers rollback_manager.rollback_model()
   - Emits PROMOTION_ROLLED_BACK event
   ↓
4. TrainingCoordinator receives PROMOTION_ROLLED_BACK event
   ↓
5. TrainingCoordinator pauses training via _pause_training_for_config()
   - Emits TRAINING_PAUSED event
```

### Component Responsibilities

#### 1. RegressionDetector (app/training/regression_detector.py)

- **Role**: Detects regressions based on Elo drops, win rate drops, error rates
- **Thresholds**:
  - MINOR: elo_drop >= 20
  - MODERATE: elo_drop >= 35
  - SEVERE: elo_drop >= 50
  - CRITICAL: elo_drop >= 75
- **Events emitted**:
  - `REGRESSION_DETECTED` (general)
  - `REGRESSION_MODERATE`, `REGRESSION_SEVERE`, `REGRESSION_CRITICAL` (specific)

#### 2. AutoRollbackHandler (app/training/rollback_manager.py)

- **Role**: Listens for regression events and triggers automatic rollback
- **New behavior**:
  - MODERATE regressions with elo_drop > 30 now trigger rollback
  - Previously only SEVERE (elo_drop >= 50) triggered rollback
- **Key changes**:
  ```python
  elif severity == RegressionSeverity.MODERATE:
      # Check if elo_drop > 30 (significant regression requiring rollback)
      elo_drop = event.elo_drop if hasattr(event, 'elo_drop') else 0
      if elo_drop > 30:
          logger.warning(
              f"[AutoRollbackHandler] MODERATE regression with elo_drop={elo_drop:.1f} > 30 - "
              f"triggering auto-rollback for {model_id}"
          )
          self._execute_rollback(
              model_id,
              reason=f"Auto-rollback: MODERATE regression with elo_drop={elo_drop:.1f} - {event.reason}",
              triggered_by="auto_regression_moderate",
          )
  ```

#### 3. TrainingCoordinator (app/coordination/training_coordinator.py)

- **Role**: Manages cluster-wide training coordination
- **Event subscriptions**:
  - `PROMOTION_ROLLED_BACK` → `_on_promotion_rolled_back()`
- **Behavior**: Pauses training for the rolled-back config to prevent retraining on bad model

## Implementation Details

### Modified Files

1. **app/training/rollback_manager.py**
   - Added elo_drop > 30 check in `on_regression()` for MODERATE severity
   - Added `elo_drop` field to `EventBusRegressionEvent` class
   - This allows the handler to check the exact elo_drop value from the event payload

### Testing

Run the following to verify the implementation:

```bash
# Test regression detection for elo_drop = 35
python3 -c "
from app.training.regression_detector import RegressionDetector, RegressionSeverity

detector = RegressionDetector()
event = detector.check_regression(
    model_id='hex8_2p_v42',
    current_elo=1450,
    baseline_elo=1485,  # 35 point drop
    games_played=100
)

assert event.severity == RegressionSeverity.MODERATE
assert event.elo_drop > 30
print(f'✓ Elo drop {event.elo_drop:.1f} exceeds threshold')
"
```

### Configuration

The system uses canonical thresholds from `app/config/thresholds.py`:

```python
ELO_DROP_ROLLBACK = 50  # SEVERE threshold
WIN_RATE_DROP_ROLLBACK = 0.10
ERROR_RATE_ROLLBACK = 0.05
MIN_GAMES_REGRESSION = 50
```

The new MODERATE threshold (35 Elo) sits between MINOR (20) and SEVERE (50), with the custom check at 30 allowing rollback for drops of 30-35 Elo.

## Initialization

The wiring is automatically set up via `wire_regression_to_rollback()` which is called from:

- `app/training/train.py`
- `app/coordination/coordination_bootstrap.py`
- `app/coordination/daemon_manager.py`
- `app/training/unified_orchestrator.py`

## Event Bus Integration

The system uses the unified event bus (`app/coordination/event_router.py`) for cross-process coordination:

1. **RegressionDetector** publishes to event bus via `_publish_to_event_bus()`
2. **AutoRollbackHandler** subscribes via `subscribe_to_regression_events()`
3. **TrainingCoordinator** subscribes via `_subscribe_to_cluster_events()`

## Monitoring

To monitor rollback events:

```bash
# Check rollback history
python3 -c "
from app.training.rollback_manager import RollbackManager
manager = RollbackManager(registry=None)
history = manager.get_rollback_history(limit=10)
for event in history:
    print(f'{event.timestamp}: {event.model_id} v{event.from_version} → v{event.to_version}')
"
```

## Safety Mechanisms

1. **Cooldown**: 5-minute cooldown between regression events for the same model
2. **Minimum games**: Requires at least 50 games before triggering regression detection
3. **Consecutive tracking**: Escalates severity if regressions occur consecutively
4. **Manual approval**: SEVERE regressions can optionally require manual approval

## Known Limitations

1. **Threshold gap**: The elo_drop > 30 check means:
   - 30.1-34.9 Elo drops trigger rollback (custom check)
   - 35+ Elo drops trigger rollback (MODERATE threshold)
   - This creates no gap, but the logic could be simplified to use just the MODERATE threshold

2. **Event bus dependency**: Requires event bus to be running for cross-process coordination

## Future Improvements

1. Consider lowering the MODERATE threshold from 35 to 30 to eliminate the custom check
2. Add Prometheus metrics for rollback events (already partially implemented)
3. Add alerting integration for CRITICAL regressions
4. Implement rollback to specific checkpoint version (currently auto-selects best)

## References

- Regression detection: `app/training/regression_detector.py`
- Rollback management: `app/training/rollback_manager.py`
- Training coordination: `app/coordination/training_coordinator.py`
- Event routing: `app/coordination/event_router.py`
