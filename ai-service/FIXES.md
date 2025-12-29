# Recent Fixes

> Status: Historical snapshot (Dec 2025). For current issues and changes, see
> `ai-service/docs/KNOWN_ISSUES.md` and `CHANGELOG.md`.

## Model Distribution Race Condition (Dec 26, 2025)

**Problem:**
Selfplay nodes would receive `MODEL_PROMOTED` events and immediately try to load the new model, but the `ModelDistributionDaemon` hadn't finished distributing the model to the cluster yet. This caused selfplay to fail with "model not found" errors.

**Race Condition Timeline:**

1. Training completes and promotes model → emits `MODEL_PROMOTED` event
2. `ModelDistributionDaemon` receives event and starts distributing (takes 30-300s)
3. Selfplay nodes receive `MODEL_PROMOTED` event and try to load model **immediately**
4. **Model doesn't exist yet** → selfplay fails

**Solution:**
Added a `wait_for_model_distribution()` mechanism that:

1. Checks if model already exists locally (fast path)
2. If not, subscribes to `MODEL_DISTRIBUTION_COMPLETE` event
3. Waits up to 300 seconds for distribution to complete
4. Falls back gracefully if distribution times out

**Implementation:**

- `app/coordination/model_distribution_daemon.py`:
  - Added `wait_for_model_distribution()` async function
  - Added `check_model_availability()` sync function
  - Added `MODEL_DISTRIBUTION_COMPLETE` event type to `data_events.py`

- `app/training/selfplay_runner.py`:
  - Modified `_load_model()` to call `_wait_for_model_availability()` first
  - Handles both async and sync contexts correctly

**Testing:**

```python
# Test 1: Model exists - returns immediately
assert check_model_availability('hex8', 2) == True

# Test 2: Model doesn't exist - times out correctly
result = await wait_for_model_distribution('nonexistent', 99, timeout=2.0)
assert result == False  # Correctly timed out
```

**Files Modified:**

- `app/coordination/model_distribution_daemon.py` - Added wait functions
- `app/training/selfplay_runner.py` - Integrated wait in model loading
- `app/distributed/data_events.py` - Added `MODEL_DISTRIBUTION_COMPLETE` event type
