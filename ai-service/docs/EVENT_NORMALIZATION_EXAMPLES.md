# Event Normalization - Before/After Examples

This document shows concrete examples of how event naming standardization works in practice.

## Quick Reference

| Before (Non-Canonical) | After (Canonical) | Auto-Normalized? |
|------------------------|-------------------|------------------|
| `sync_complete` | `DATA_SYNC_COMPLETED` | ✅ Yes |
| `SYNC_COMPLETE` | `DATA_SYNC_COMPLETED` | ✅ Yes |
| `CLUSTER_SYNC_COMPLETE` | `DATA_SYNC_COMPLETED` | ✅ Yes |
| `training_complete` | `TRAINING_COMPLETED` | ✅ Yes |
| `TRAINING_COMPLETE` | `TRAINING_COMPLETED` | ✅ Yes |
| `selfplay_batch_complete` | `SELFPLAY_COMPLETE` | ✅ Yes |
| `model_sync_complete` | `P2P_MODEL_SYNCED` | ✅ Yes |
| `promotion_complete` | `MODEL_PROMOTED` | ✅ Yes |
| `evaluation_complete` | `EVALUATION_COMPLETED` | ✅ Yes |

## Code Examples

### Example 1: Publishing Events

#### Before (Multiple Inconsistent Names)
```python
# Different parts of codebase using different names for same event
# File 1: auto_sync_daemon.py
await router.publish("sync_complete", payload)

# File 2: sync_coordinator.py
await router.publish("SYNC_COMPLETE", payload)

# File 3: data_pipeline_orchestrator.py
await router.publish("CLUSTER_SYNC_COMPLETE", payload)

# Result: Subscribers confused about which event to listen for
```

#### After (Normalized to Canonical)
```python
# All variants automatically normalized to canonical form
# File 1: auto_sync_daemon.py
await router.publish("sync_complete", payload)
# → Router normalizes to "DATA_SYNC_COMPLETED"

# File 2: sync_coordinator.py
await router.publish("SYNC_COMPLETE", payload)
# → Router normalizes to "DATA_SYNC_COMPLETED"

# File 3: data_pipeline_orchestrator.py
await router.publish("CLUSTER_SYNC_COMPLETE", payload)
# → Router normalizes to "DATA_SYNC_COMPLETED"

# Result: All subscribers receive "DATA_SYNC_COMPLETED"
```

### Example 2: Subscribing to Events

#### Before (Had to Subscribe to Multiple Variants)
```python
from app.coordination.event_router import get_router

router = get_router()

# Need to subscribe to all possible names
router.subscribe("sync_complete", on_sync)
router.subscribe("SYNC_COMPLETE", on_sync)
router.subscribe("CLUSTER_SYNC_COMPLETE", on_sync)
router.subscribe("DATA_SYNC_COMPLETED", on_sync)

# Result: Brittle, easy to miss a variant
```

#### After (Single Canonical Subscription)
```python
from app.coordination.event_router import get_router

router = get_router()

# Subscribe once to canonical name
router.subscribe("DATA_SYNC_COMPLETED", on_sync)

# Receives events published as ANY of:
# - sync_complete
# - SYNC_COMPLETE
# - CLUSTER_SYNC_COMPLETE
# - DATA_SYNC_COMPLETED

# Result: Simple, reliable, complete
```

### Example 3: Event Emission in Daemons

#### Before (Inconsistent Event Names)
```python
# auto_sync_daemon.py
async def _emit_sync_completed(self, games_synced: int):
    await router.publish(
        event_type=DataEventType.DATA_SYNC_COMPLETED,  # Uses COMPLETED
        payload={"games_synced": games_synced}
    )

# sync_coordinator.py
async def _emit_sync_complete(self):
    await router.publish(
        event_type=StageEvent.SYNC_COMPLETE,  # Uses COMPLETE
        payload={}
    )

# Result: Two events that mean the same thing but named differently
```

#### After (Both Normalized)
```python
# auto_sync_daemon.py
async def _emit_sync_completed(self, games_synced: int):
    await router.publish(
        event_type=DataEventType.DATA_SYNC_COMPLETED,
        payload={"games_synced": games_synced}
    )
    # → Router publishes as "DATA_SYNC_COMPLETED"

# sync_coordinator.py
async def _emit_sync_complete(self):
    await router.publish(
        event_type=StageEvent.SYNC_COMPLETE,
        payload={}
    )
    # → Router normalizes to "DATA_SYNC_COMPLETED"

# Result: Both emit the same canonical event type
```

### Example 4: Event History and Debugging

#### Before (Mixed Names in Logs)
```
[EventRouter] Published: sync_complete
[EventRouter] Published: SYNC_COMPLETE
[EventRouter] Published: CLUSTER_SYNC_COMPLETE
[EventRouter] Published: DATA_SYNC_COMPLETED

# Hard to trace: are these 4 different event types or the same?
```

#### After (Canonical Names in Logs)
```
[EventRouter] Normalized 'sync_complete' → 'DATA_SYNC_COMPLETED'
[EventRouter] Published: DATA_SYNC_COMPLETED
[EventRouter] Normalized 'SYNC_COMPLETE' → 'DATA_SYNC_COMPLETED'
[EventRouter] Published: DATA_SYNC_COMPLETED
[EventRouter] Normalized 'CLUSTER_SYNC_COMPLETE' → 'DATA_SYNC_COMPLETED'
[EventRouter] Published: DATA_SYNC_COMPLETED
[EventRouter] Published: DATA_SYNC_COMPLETED

# Clear: all the same event type, normalized for consistency
```

### Example 5: Auditing Event Usage

#### Before (Unknown Usage Patterns)
```python
# No way to know which event names are used in production
# Manual grep required: grep -r "sync_complete\|SYNC_COMPLETE" .
```

#### After (Programmatic Audit)
```python
from app.coordination.event_normalization import audit_event_usage
from app.coordination.event_router import get_router

# Get event history
router = get_router()
history = [e.event_type for e in router.get_history(limit=1000)]

# Audit usage
audit = audit_event_usage(history)

print(f"Total events: {audit['total_events']}")
print(f"Non-canonical: {audit['non_canonical_count']} ({audit['normalization_rate']:.1%})")

print("\nNon-canonical variants in use:")
for variant, canonical in audit['non_canonical_variants'].items():
    count = history.count(variant)
    print(f"  {variant:30s} → {canonical:30s} ({count} occurrences)")

# Output:
# Total events: 1000
# Non-canonical: 623 (62.3%)
#
# Non-canonical variants in use:
#   sync_complete              → DATA_SYNC_COMPLETED         (245 occurrences)
#   SYNC_COMPLETE              → DATA_SYNC_COMPLETED         (189 occurrences)
#   training_complete          → TRAINING_COMPLETED          (156 occurrences)
#   selfplay_batch_complete    → SELFPLAY_COMPLETE           (33 occurrences)
```

## Real-World Migration Examples

### Migration Example 1: Auto-Sync Daemon

#### Before
```python
# app/coordination/auto_sync_daemon.py
async def _emit_sync_completed(self, games_synced: int):
    """Emit DATA_SYNC_COMPLETED event."""
    try:
        from app.coordination.event_router import get_router
        from app.distributed.data_events import DataEventType

        router = get_router()
        if router:
            await router.publish(
                event_type=DataEventType.DATA_SYNC_COMPLETED,
                payload={
                    "node_id": self.node_id,
                    "games_synced": games_synced,
                }
            )
    except Exception as e:
        logger.debug(f"Could not emit DATA_SYNC_COMPLETED: {e}")
```

#### After (No Changes Required!)
```python
# Same code - router automatically handles normalization
# No migration needed, continues working as-is
```

### Migration Example 2: Sync Coordinator

#### Before
```python
# app/coordination/sync_coordinator.py
async def on_sync_complete(result):
    """Handle SYNC_COMPLETE - update scheduler state."""
    games = result.payload.get("games_synced", 0)
    # ...

router.subscribe("SYNC_COMPLETE", on_sync_complete)
router.subscribe("DATA_SYNC_COMPLETED", on_sync_complete)  # Also subscribe to variant
```

#### After (Simplified)
```python
# app/coordination/sync_coordinator.py
async def on_sync_complete(result):
    """Handle DATA_SYNC_COMPLETED - update scheduler state."""
    games = result.payload.get("games_synced", 0)
    # ...

# Only need to subscribe to canonical name
router.subscribe("DATA_SYNC_COMPLETED", on_sync_complete)

# Receives events from:
# - SYNC_COMPLETE
# - sync_complete
# - DATA_SYNC_COMPLETED
# - cluster_sync_complete
```

### Migration Example 3: Event Mappings

#### Before
```python
# app/coordination/event_mappings.py
STAGE_TO_DATA_EVENT_MAP = {
    "sync_complete": "sync_completed",  # Inconsistent tense
    "cluster_sync_complete": "sync_completed",  # Duplicate mapping
    "training_complete": "training_completed",  # Inconsistent suffix
    "evaluation_complete": "evaluation_completed",
}
```

#### After (Canonical)
```python
# app/coordination/event_mappings.py
STAGE_TO_DATA_EVENT_MAP = {
    "sync_complete": "DATA_SYNC_COMPLETED",  # Canonical
    "cluster_sync_complete": "DATA_SYNC_COMPLETED",  # Canonical
    "training_complete": "TRAINING_COMPLETED",  # Canonical
    "evaluation_complete": "EVALUATION_COMPLETED",  # Canonical
}

# Router further normalizes these when publishing
```

## Testing Examples

### Test Before
```python
def test_sync_complete():
    """Test that sync complete event is handled."""
    # Which event name should we test?
    # sync_complete? SYNC_COMPLETE? DATA_SYNC_COMPLETED?

    # End up testing multiple variants
    for event_name in ["sync_complete", "SYNC_COMPLETE", "DATA_SYNC_COMPLETED"]:
        router.publish(event_name, {"host": "test"})
        # ...assert...
```

### Test After
```python
def test_sync_complete():
    """Test that DATA_SYNC_COMPLETED event is handled."""
    # Test canonical name
    router.publish("DATA_SYNC_COMPLETED", {"host": "test"})
    # ...assert...

    # Optional: Test that variants are normalized
    assert normalize_event_type("sync_complete") == "DATA_SYNC_COMPLETED"
    assert normalize_event_type("SYNC_COMPLETE") == "DATA_SYNC_COMPLETED"
```

## Common Patterns

### Pattern 1: Lifecycle Events

#### Before (Inconsistent)
```python
# Training lifecycle with inconsistent naming
"training_start"      # present tense
"TRAINING_COMPLETE"   # all caps, incomplete tense
"training_failed"     # lowercase, complete tense
```

#### After (Canonical)
```python
# Training lifecycle with consistent naming
"TRAINING_STARTED"    # past tense, UPPERCASE
"TRAINING_COMPLETED"  # past tense, UPPERCASE, complete
"TRAINING_FAILED"     # past tense, UPPERCASE
```

### Pattern 2: Sync Events

#### Before (Ambiguous)
```python
# What kind of sync?
"SYNC_COMPLETE"       # Data? Model? Cluster?
"sync_started"        # lowercase vs uppercase
"SYNC_FAILED"         # uppercase
```

#### After (Specific)
```python
# Clear subject + action + state
"DATA_SYNC_STARTED"      # Data synchronization started
"DATA_SYNC_COMPLETED"    # Data synchronization completed
"DATA_SYNC_FAILED"       # Data synchronization failed
"P2P_MODEL_SYNCED"       # Model distributed via P2P
```

### Pattern 3: Completion Events

#### Before (Multiple Forms)
```python
"training_complete"       # adjective form
"TRAINING_COMPLETED"      # past tense
"promotion_complete"      # adjective form
"EVALUATION_COMPLETE"     # adjective form
```

#### After (Consistent Tense)
```python
"TRAINING_COMPLETED"      # past tense (action completed)
"PROMOTION_COMPLETED"     # past tense → "MODEL_PROMOTED" (state change)
"EVALUATION_COMPLETED"    # past tense (action completed)
```

## Migration Checklist

### For New Code
- [ ] Use canonical event names from `EVENT_NAMING_CONVENTION.md`
- [ ] Subscribe to canonical names only
- [ ] Use `is_canonical()` to verify event names in tests

### For Existing Code (Optional)
- [ ] Update event publishers to use canonical names
- [ ] Update event subscribers to use canonical names
- [ ] Remove duplicate subscriptions for variant names
- [ ] Update test assertions to use canonical names

### For Documentation
- [ ] Update code comments to reference canonical names
- [ ] Update API documentation with canonical event types
- [ ] Add migration notes for deprecated event names

## Benefits Realized

### Before Standardization
- ❌ 87 occurrences of SYNC_COMPLETE variants across 28 files
- ❌ 124 occurrences of SELFPLAY_COMPLETE variants across 40 files
- ❌ Developers unsure which event name to use
- ❌ Tests brittle (need to test multiple variants)
- ❌ Logs confusing (same event, different names)

### After Standardization
- ✅ Single canonical name: `DATA_SYNC_COMPLETED`
- ✅ Single canonical name: `SELFPLAY_COMPLETE`
- ✅ Clear naming convention for new events
- ✅ Tests simple (test canonical name only)
- ✅ Logs clear (all normalized to canonical)
- ✅ Zero breaking changes (backward compatible)

## See Also

- [Event Naming Convention](EVENT_NAMING_CONVENTION.md) - Complete naming rules
- [Event Normalization Summary](EVENT_NORMALIZATION_SUMMARY.md) - Implementation details
- [Event Router](/Users/armand/Development/RingRift/ai-service/app/coordination/event_router.py) - Normalization integration
- [Event Normalization](/Users/armand/Development/RingRift/ai-service/app/coordination/event_normalization.py) - Normalization utilities
