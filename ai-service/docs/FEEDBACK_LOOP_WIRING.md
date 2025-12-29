# Feedback Loop Wiring: Elo Momentum → Selfplay Rate

## Status: ✅ FULLY IMPLEMENTED (December 2025 - Phase 19)

This document confirms that the critical feedback loop connecting Elo momentum to selfplay allocation is **already fully wired and functional**.

## Problem Statement

The original concern was that `FeedbackAccelerator.get_selfplay_multiplier()` was defined but not being called by `selfplay_scheduler.py`.

## Solution: Already Implemented

**The feedback loop has been fully wired since December 2025 (Phase 19).**

### Implementation Details

The complete flow is:

1. **Elo Update → Momentum Calculation**
   - Location: `app/training/feedback_accelerator.py:461-516`
   - Method: `FeedbackAccelerator.record_elo_update()`
   - Tracks Elo history and calculates momentum states

2. **Momentum → Selfplay Multiplier**
   - Location: `app/training/feedback_accelerator.py:787-834`
   - Method: `FeedbackAccelerator.get_selfplay_multiplier()`
   - Maps momentum states to multipliers:
     - `ACCELERATING`: 1.5x (capitalize on positive momentum)
     - `IMPROVING`: 1.25x (boost for continued improvement)
     - `STABLE`: 1.0x (normal rate)
     - `PLATEAU`: 1.1x (slight boost to break plateau)
     - `REGRESSING`: 0.75x (reduce noise, focus on quality)

3. **Scheduler Retrieves Multipliers**
   - Location: `app/coordination/selfplay_scheduler.py:605-641`
   - Method: `SelfplayScheduler._get_momentum_multipliers()`
   - **Line 627**: Calls `accelerator.get_selfplay_multiplier(config_key)` ✅
   - Returns dict mapping config_key → multiplier

4. **Priority Update Loop**
   - Location: `app/coordination/selfplay_scheduler.py:281-346`
   - Method: `SelfplayScheduler._update_priorities()`
   - **Line 305**: Calls `_get_momentum_multipliers()` ✅
   - **Line 334-335**: Stores result in `priority.momentum_multiplier` ✅

5. **Priority Score Calculation**
   - Location: `app/coordination/selfplay_scheduler.py:375-428`
   - Method: `SelfplayScheduler._compute_priority_score()`
   - **Line 420**: Applies momentum multiplier to score ✅
   ```python
   score *= priority.momentum_multiplier
   ```

## Verification

Run the included test scripts to verify the wiring:

```bash
# Basic wiring test
python test_feedback_wiring.py

# Integration test (demonstrates full feedback loop)
python test_feedback_loop_integration.py
```

### Test Results (Dec 26, 2025)

```
✅ ALL TESTS PASSED - Feedback loop is correctly wired!

TEST SUMMARY
✓ PASS: FeedbackAccelerator.get_selfplay_multiplier()
✓ PASS: SelfplayScheduler._get_momentum_multipliers()
✓ PASS: Priority Score Calculation
```

## How It Works

### Example: Accelerating Config

1. Model achieves +30 Elo over 5 evaluations
2. `FeedbackAccelerator` detects `ACCELERATING` momentum state
3. `get_selfplay_multiplier("hex8_2p")` returns `1.5`
4. `SelfplayScheduler` retrieves this via `_get_momentum_multipliers()`
5. Priority score is multiplied by 1.5x
6. Config gets 50% more selfplay games allocated

### Example: Regressing Config

1. Model loses -25 Elo over 5 evaluations
2. `FeedbackAccelerator` detects `REGRESSING` momentum state
3. `get_selfplay_multiplier("square8_2p")` returns `0.75`
4. `SelfplayScheduler` retrieves this via `_get_momentum_multipliers()`
5. Priority score is multiplied by 0.75x
6. Config gets 25% fewer selfplay games allocated

## Additional Multipliers

The priority score calculation also includes:

- **Curriculum weight** (Phase 2C.3): Boosts configs needing more training data
- **Improvement boost** (Phase 5): From `ImprovementOptimizer` for promotion streaks
- **Exploration boost**: Dynamic exploration factor from feedback signals (quality-driven via `EXPLORATION_ADJUSTED`)
- **Data deficit factor**: Boosts configs with low game counts (especially large boards)
- **Priority overrides**: From `unified_loop.yaml` for critical data-starved configs

## Event Flow

The feedback loop also emits events for monitoring:

1. **SELFPLAY_RATE_CHANGED** (Phase 19.3)
   - Emitted by `FeedbackAccelerator` when multiplier changes >20%
   - Location: `feedback_accelerator.py:836-903`
   - Subscribed by: `SelfplayScheduler._on_selfplay_rate_changed()` (line 1069)

2. **SELFPLAY_TARGET_UPDATED**
   - Emitted by `SelfplayScheduler` when priorities change
   - Used by downstream systems (e.g., `IdleResourceDaemon`)

3. **EXPLORATION_ADJUSTED**
   - Emitted by `FeedbackLoopController` when quality score trends change
   - Subscribed by: `SelfplayScheduler` (adjusts exploration parameters)

## Code References

### FeedbackAccelerator

```python
# File: app/training/feedback_accelerator.py

def get_selfplay_multiplier(self, config_key: str) -> float:
    """Get selfplay games multiplier based on Elo momentum (December 2025)."""
    momentum = self._configs.get(config_key)
    if not momentum:
        return 1.0

    multiplier_map = {
        MomentumState.ACCELERATING: 1.5,
        MomentumState.IMPROVING: 1.25,
        MomentumState.STABLE: 1.0,
        MomentumState.PLATEAU: 1.1,
        MomentumState.REGRESSING: 0.75,
    }

    base_multiplier = multiplier_map.get(momentum.momentum_state, 1.0)

    # Adjustments for consecutive patterns
    if momentum.consecutive_improvements >= 3:
        base_multiplier = min(base_multiplier * 1.1, 1.5)
    if momentum.consecutive_plateaus >= 3:
        base_multiplier = max(base_multiplier * 0.9, 0.5)

    return base_multiplier
```

### SelfplayScheduler

```python
# File: app/coordination/selfplay_scheduler.py

def _get_momentum_multipliers(self) -> dict[str, float]:
    """Get momentum multipliers from FeedbackAccelerator per config."""
    result: dict[str, float] = {}

    from app.training.feedback_accelerator import get_feedback_accelerator
    accelerator = get_feedback_accelerator()

    for config_key in ALL_CONFIGS:
        multiplier = accelerator.get_selfplay_multiplier(config_key)  # ✅ THE CALL
        result[config_key] = multiplier

    return result

def _compute_priority_score(self, priority: ConfigPriority) -> float:
    """Compute overall priority score for a configuration."""
    # ... calculate base score ...

    # Phase 19: Apply momentum multiplier from FeedbackAccelerator
    score *= priority.momentum_multiplier  # ✅ THE APPLICATION

    return score
```

## Conclusion

The feedback loop is **fully functional and has been since December 2025**. No changes are needed.

The wiring correctly implements the Elo momentum → Selfplay rate coupling:

- ✅ `get_selfplay_multiplier()` is defined
- ✅ `_get_momentum_multipliers()` calls it
- ✅ `_update_priorities()` retrieves the results
- ✅ `_compute_priority_score()` applies them to the priority calculation
- ✅ Accelerating configs get higher selfplay priority
- ✅ Regressing configs get lower selfplay priority

## Next Steps

If you want to enhance the feedback loop further, consider:

1. **Monitoring**: Add dashboard visualization for momentum multipliers
2. **Tuning**: Adjust multiplier values based on cluster performance data
3. **Expansion**: Add more momentum states (e.g., "stagnant", "volatile")
4. **Decay**: Implement time-based decay for temporary boosts

But the core wiring is **complete and operational**.
