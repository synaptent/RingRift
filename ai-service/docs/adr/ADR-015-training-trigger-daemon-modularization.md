# ADR-015: Training Trigger Daemon Modularization

**Status**: Accepted
**Date**: January 2026
**Author**: Claude Code (Phase 2 Modularization)

## Context

The `training_trigger_daemon.py` file grew to 4,037 lines (~46,800 tokens), exceeding Claude's context limit of ~25,000 tokens. This made it impossible to read the entire file at once, complicating debugging, maintenance, and AI-assisted development.

The monolithic structure contained multiple responsibilities:

- Training threshold monitoring
- Architecture selection for training
- Velocity-based parameter adjustment
- Retry and failure recovery logic
- Event subscriptions and emissions

The file was one of two identified "large file" problems (the other being `data_events.py` at 4,117 lines).

## Decision

Extract three focused modules from `training_trigger_daemon.py`:

1. **`training_architecture_selector.py`** (~150 lines)
   - `get_training_params_for_intensity()` - maps intensity to epochs/batch/LR
   - `select_architecture_for_training()` - weighted architecture selection
   - `apply_velocity_amplification()` - adjusts params based on Elo velocity

2. **`training_retry_manager.py`** (~200 lines)
   - `TrainingRetryState` - tracks retry attempts per config
   - `RetryDecision` - enum for retry verdicts
   - `should_retry_training()` - retry eligibility check
   - `get_retry_backoff()` - exponential backoff calculation

3. **`training_state_tracker.py`** (~200 lines)
   - `TrainingStateTracker` - singleton for in-flight training tracking
   - `TrainingJobState` - per-job state (started, config, timestamps)
   - `is_training_active()` - check if training is in progress
   - `get_active_training_count()` - count active training jobs

The main `training_trigger_daemon.py` now imports and delegates to these modules, reducing its core logic while maintaining the same public API.

## Consequences

### Positive

1. **Context-aware editing**: Each module fits within AI context limits
2. **Clear separation of concerns**: Architecture selection, retry logic, and state tracking are independently testable
3. **Reduced merge conflicts**: Changes to retry logic don't touch architecture selection
4. **Improved discoverability**: Functions are easier to find in smaller, focused files
5. **Better test isolation**: Unit tests can target specific modules without daemon overhead

### Negative

1. **Additional imports**: `training_trigger_daemon.py` now imports 3 new modules
2. **Potential circular imports**: Careful ordering required (mitigated by TYPE_CHECKING imports)
3. **Migration overhead**: Existing internal references needed updating

## Implementation Notes

### Module Structure

```
app/coordination/
├── training_trigger_daemon.py      # Main daemon (reduced from 4,037 to ~3,200 lines)
├── training_architecture_selector.py  # Architecture and param selection
├── training_retry_manager.py          # Retry logic and state
└── training_state_tracker.py          # Training job tracking
```

### Test Coverage

Each extracted module has dedicated tests:

- `tests/unit/coordination/test_training_architecture_selector.py` (28 tests)
- `tests/unit/coordination/test_training_retry_manager.py` (25 tests)
- `tests/unit/coordination/test_training_state_tracker.py` (22 tests)

### Backward Compatibility

The public API of `training_trigger_daemon.py` is unchanged:

- `TrainingTriggerDaemon` class
- Event subscriptions (TRAINING_THRESHOLD_REACHED, etc.)
- Configuration via environment variables

Internal consumers that imported helper functions directly from `training_trigger_daemon.py` should migrate to the new modules, but the daemon itself handles all orchestration.

## Related ADRs

- [ADR-001](ADR-001-event-driven-architecture.md) - Event-driven architecture (daemon subscribes to events)
- [ADR-002](ADR-002-daemon-lifecycle-management.md) - Daemon lifecycle management
- [ADR-004](ADR-004-quality-gate-feedback-loop.md) - Quality gates that trigger training

## Future Work

1. **data_events.py modularization**: The other large file (~4,117 lines) remains monolithic. Similar extraction of emit\_\* functions into domain-specific modules is planned.

2. **Further training_trigger_daemon.py reduction**: Additional extraction candidates include:
   - Quality gate evaluation logic
   - NPZ validation and filtering
   - Cluster node selection for training
