# RingRift Event Naming Convention

**Status**: Active (December 2025)
**Purpose**: Standardize event naming across the RingRift AI service event system

## Overview

The RingRift event system uses a canonical naming convention to ensure consistency across three event buses:
1. **DataEventBus** (`data_events.py`) - In-memory async event bus
2. **StageEventBus** (`stage_events.py`) - Pipeline stage completion events
3. **CrossProcessEventQueue** (`cross_process_events.py`) - SQLite-backed IPC

This document defines the canonical naming rules and provides guidance for naming new events.

## Canonical Format

```
{SUBJECT}_{ACTION}_{MODIFIER}
```

### Components

- **SUBJECT**: What the event is about (required)
  - Examples: `DATA`, `MODEL`, `TRAINING`, `EVALUATION`, `SELFPLAY`

- **ACTION**: What action occurred (required)
  - Examples: `SYNC`, `TRAIN`, `EVALUATE`, `PROMOTE`, `EXPORT`

- **MODIFIER**: Optional qualifier for specificity
  - Examples: `BATCH`, `GPU`, `CANONICAL`, `COMPLETE`

### Examples

| Canonical Name | Subject | Action | Modifier | Description |
|----------------|---------|--------|----------|-------------|
| `DATA_SYNC_COMPLETED` | DATA | SYNC | COMPLETED | Data synchronization finished |
| `TRAINING_STARTED` | TRAINING | - | STARTED | Training job started |
| `MODEL_PROMOTED` | MODEL | PROMOTE | - | Model promoted to production |
| `SELFPLAY_COMPLETE` | SELFPLAY | - | COMPLETE | Selfplay batch finished |
| `NPZ_EXPORT_COMPLETE` | NPZ | EXPORT | COMPLETE | NPZ export finished |

## Tense Rules

### Completion Events
Use past tense with `_COMPLETED` suffix:
- `TRAINING_COMPLETED` ✓
- `TRAINING_COMPLETE` ✗ (ambiguous)
- `TRAINING_DONE` ✗ (non-standard)

### Start Events
Use past tense with `_STARTED` suffix:
- `EVALUATION_STARTED` ✓
- `EVALUATION_START` ✗ (present tense)
- `EVALUATION_BEGIN` ✗ (non-standard)

### State Change Events
Use past participle:
- `MODEL_PROMOTED` ✓ (state change)
- `PROMOTION_COMPLETE` ✗ (use subject-first)
- `P2P_MODEL_SYNCED` ✓ (state change)

### Failure Events
Use past tense with `_FAILED` suffix:
- `TRAINING_FAILED` ✓
- `TRAINING_ERROR` ✗ (ambiguous)
- `TRAINING_FAIL` ✗ (present tense)

## Specificity Rules

### 1. Always Include Subject
Be explicit about what the event pertains to:

| ✗ Avoid | ✓ Use Instead | Reason |
|---------|---------------|---------|
| `SYNC_COMPLETE` | `DATA_SYNC_COMPLETED` | Ambiguous: what kind of sync? |
| `COMPLETE` | `TRAINING_COMPLETED` | Incomplete: what completed? |
| `STARTED` | `EVALUATION_STARTED` | Missing context |

### 2. Be Explicit About Scope
Distinguish between batch and aggregate events:

| ✗ Avoid | ✓ Use Instead | Context |
|---------|---------------|---------|
| `COMPLETE` | `SELFPLAY_COMPLETE` | Single batch completion |
| `GAMES_AVAILABLE` | `NEW_GAMES_AVAILABLE` | New games from selfplay |
| `SYNC` | `DATA_SYNC_COMPLETED` | Full data sync finished |

### 3. Distinguish Between Stages
Use different names for different lifecycle stages:

| Event | Stage | Purpose |
|-------|-------|---------|
| `TRAINING_STARTED` | Start | Training job began |
| `TRAINING_PROGRESS` | Progress | Training checkpoint |
| `TRAINING_COMPLETED` | Completion | Training job finished |
| `TRAINING_FAILED` | Failure | Training job failed |

## Case Conventions

### Cross-Process Events (Canonical)
All cross-process events use `UPPERCASE_SNAKE_CASE`:
```python
# Correct
"DATA_SYNC_COMPLETED"
"TRAINING_STARTED"
"MODEL_PROMOTED"

# Incorrect
"data_sync_completed"  # lowercase
"DataSyncCompleted"    # PascalCase
"dataSyncCompleted"    # camelCase
```

### Stage Events (Internal)
Stage events use `lowercase_snake_case`:
```python
# StageEvent enum values
StageEvent.SYNC_COMPLETE = "sync_complete"
StageEvent.TRAINING_COMPLETE = "training_complete"

# Note: Router normalizes these to canonical UPPERCASE form
```

### Data Events (Internal)
Data events match `DataEventType` enum values:
```python
# DataEventType enum values
DataEventType.DATA_SYNC_COMPLETED = "sync_completed"
DataEventType.TRAINING_STARTED = "training_started"
```

## Common Event Patterns

### 1. Lifecycle Events (Start → Complete → Fail)

```
SUBJECT_STARTED      # Action initiated
SUBJECT_PROGRESS     # Progress update (optional)
SUBJECT_COMPLETED    # Action succeeded
SUBJECT_FAILED       # Action failed
```

Example: Training lifecycle
```
TRAINING_STARTED
TRAINING_PROGRESS
TRAINING_COMPLETED
TRAINING_FAILED
```

### 2. Sync Events (Start → Complete → Fail)

```
{TYPE}_SYNC_STARTED
{TYPE}_SYNC_COMPLETED
{TYPE}_SYNC_FAILED
```

Example: Data sync lifecycle
```
DATA_SYNC_STARTED
DATA_SYNC_COMPLETED
DATA_SYNC_FAILED
```

### 3. State Change Events

```
SUBJECT_{NEW_STATE}
```

Example: Model promotion
```
MODEL_PROMOTED       # Model reached production state
TIER_PROMOTION       # Model advanced difficulty tier
```

## Deprecated Forms (Avoid)

The following forms are automatically normalized by the event router but should be updated in new code:

| Deprecated | Canonical | Update Priority |
|------------|-----------|-----------------|
| `SYNC_COMPLETE` | `DATA_SYNC_COMPLETED` | High |
| `TRAINING_COMPLETE` | `TRAINING_COMPLETED` | High |
| `PROMOTION_COMPLETE` | `MODEL_PROMOTED` | Medium |
| `CLUSTER_SYNC_COMPLETE` | `DATA_SYNC_COMPLETED` | Medium |
| `MODEL_SYNC_COMPLETE` | `P2P_MODEL_SYNCED` | Medium |
| `EVALUATION_COMPLETE` | `EVALUATION_COMPLETED` | Low (still clear) |

## Migration Path

### Automatic Normalization
All events published through the event router are automatically normalized:

```python
from app.coordination.event_router import publish

# Legacy code (still works)
await publish("SYNC_COMPLETE", payload)

# Router automatically normalizes to:
# "DATA_SYNC_COMPLETED"

# Subscribers receive canonical name
router.subscribe("DATA_SYNC_COMPLETED", handler)
```

### Updating Code
Update code gradually to use canonical names:

1. **New code**: Use canonical names from the start
2. **Existing code**: Update opportunistically during refactoring
3. **Legacy systems**: Let the router normalize (no breaking changes)

### Validation
Use the normalization utilities to audit event usage:

```python
from app.coordination.event_normalization import (
    audit_event_usage,
    normalize_event_type,
    is_canonical,
)

# Check if an event is canonical
is_canonical("DATA_SYNC_COMPLETED")  # → True
is_canonical("SYNC_COMPLETE")        # → False

# Normalize to canonical form
normalize_event_type("SYNC_COMPLETE")  # → "DATA_SYNC_COMPLETED"

# Audit historical usage
event_history = router.get_history()
audit = audit_event_usage([e.event_type for e in event_history])
print(audit["recommendations"])
```

## Guidelines for New Events

### 1. Choose the Right Subject
Pick the most specific subject that clearly identifies what the event is about:

```python
# Good: Specific subjects
"DATA_SYNC_COMPLETED"      # Data synchronization
"MODEL_PROMOTED"           # Model lifecycle
"SELFPLAY_COMPLETE"        # Selfplay batch
"NPZ_EXPORT_COMPLETE"      # NPZ file export

# Bad: Generic subjects
"PROCESS_COMPLETE"         # Too generic
"OPERATION_FINISHED"       # Unclear
"EVENT_OCCURRED"           # Meaningless
```

### 2. Use Standard Action Verbs
Prefer well-known action verbs:

| Action | Events |
|--------|--------|
| `SYNC` | `DATA_SYNC_COMPLETED`, `P2P_MODEL_SYNCED` |
| `TRAIN` | `TRAINING_STARTED`, `TRAINING_COMPLETED` |
| `EVALUATE` | `EVALUATION_STARTED`, `EVALUATION_COMPLETED` |
| `PROMOTE` | `MODEL_PROMOTED`, `TIER_PROMOTION` |
| `EXPORT` | `NPZ_EXPORT_COMPLETE` |

### 3. Add Modifiers for Disambiguation
Use modifiers when you have multiple event types for the same subject:

```python
# Different selfplay types
"SELFPLAY_COMPLETE"            # Generic
"CANONICAL_SELFPLAY_COMPLETE"  # High-quality TS engine
"GPU_SELFPLAY_COMPLETE"        # GPU-accelerated

# Different sync scopes
"DATA_SYNC_COMPLETED"          # Full data sync
"MODEL_SYNC_COMPLETE"          # Model file sync
"CLUSTER_SYNC_COMPLETE"        # Cluster-wide sync
```

### 4. Maintain Consistency with Existing Events
When adding related events, match the pattern of similar events:

```python
# If you have:
"TRAINING_STARTED"
"TRAINING_COMPLETED"
"TRAINING_FAILED"

# Add similar pattern for evaluation:
"EVALUATION_STARTED"      # ✓ Matches pattern
"EVALUATION_COMPLETED"    # ✓ Matches pattern
"EVALUATION_FAILED"       # ✓ Matches pattern

# Not:
"EVAL_START"              # ✗ Abbreviation, inconsistent
"EVALUATION_DONE"         # ✗ Different verb
"EVALUATION_ERROR"        # ✗ Different failure pattern
```

## Reference Table

### Canonical Event Types

| Category | Canonical Name | Description |
|----------|----------------|-------------|
| **Data Sync** |
| | `DATA_SYNC_STARTED` | Data synchronization started |
| | `DATA_SYNC_COMPLETED` | Data synchronization finished |
| | `DATA_SYNC_FAILED` | Data synchronization failed |
| **Model Sync** |
| | `P2P_MODEL_SYNCED` | P2P model distribution completed |
| | `MODEL_DISTRIBUTION_COMPLETE` | Model distributed to cluster |
| **Selfplay** |
| | `SELFPLAY_COMPLETE` | Selfplay batch completed |
| | `NEW_GAMES_AVAILABLE` | New games ready for training |
| **Training** |
| | `TRAINING_STARTED` | Training job started |
| | `TRAINING_COMPLETED` | Training job completed |
| | `TRAINING_FAILED` | Training job failed |
| | `TRAINING_PROGRESS` | Training progress update |
| **Evaluation** |
| | `EVALUATION_STARTED` | Evaluation started |
| | `EVALUATION_COMPLETED` | Evaluation completed |
| | `EVALUATION_FAILED` | Evaluation failed |
| **Promotion** |
| | `MODEL_PROMOTED` | Model promoted to production |
| | `PROMOTION_STARTED` | Promotion process started |
| | `PROMOTION_FAILED` | Promotion process failed |
| | `PROMOTION_REJECTED` | Model rejected for promotion |
| **Data Pipeline** |
| | `NPZ_EXPORT_COMPLETE` | NPZ export completed |
| | `PARITY_VALIDATION_COMPLETED` | Parity validation completed |
| **Optimization** |
| | `CMAES_COMPLETED` | CMA-ES optimization completed |
| | `PBT_GENERATION_COMPLETE` | PBT generation completed |
| | `NAS_COMPLETED` | NAS search completed |

## Implementation Details

### Event Normalization Module
Location: `/Users/armand/Development/RingRift/ai-service/app/coordination/event_normalization.py`

Provides:
- `normalize_event_type(event_type)` - Normalize to canonical form
- `is_canonical(event_type)` - Check if already canonical
- `get_variants(canonical_event)` - Get all known variants
- `audit_event_usage(history)` - Audit historical usage

### Event Router Integration
Location: `/Users/armand/Development/RingRift/ai-service/app/coordination/event_router.py`

The `UnifiedEventRouter.publish()` method automatically normalizes all event types before routing:

```python
async def publish(self, event_type, payload, source):
    # Extract enum value if needed
    event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)

    # Normalize to canonical form
    event_type_str = normalize_event_type(event_type_str)

    # Route normalized event
    # ...
```

## FAQ

### Q: What if I use a deprecated event name?
**A**: The event router automatically normalizes it to the canonical form. Your code will continue to work, but consider updating to the canonical name.

### Q: Can I add new event types?
**A**: Yes! Follow the naming convention and add the event to:
1. `DataEventType` enum in `data_events.py` (if it's a data event)
2. `CANONICAL_EVENT_NAMES` mapping in `event_normalization.py`

### Q: What if I disagree with a canonical name?
**A**: Open a discussion! These conventions are meant to improve consistency, not to be rigid. If you have a better suggestion, propose it with rationale.

### Q: How do I find all variants of an event?
**A**: Use the `get_variants()` function:
```python
from app.coordination.event_normalization import get_variants
variants = get_variants("DATA_SYNC_COMPLETED")
# → ["sync_complete", "SYNC_COMPLETE", "data_sync_complete", ...]
```

### Q: Do I need to update all legacy code immediately?
**A**: No! The normalization layer provides backward compatibility. Update code gradually as you touch it.

## See Also

- [`event_normalization.py`](/Users/armand/Development/RingRift/ai-service/app/coordination/event_normalization.py) - Implementation
- [`event_router.py`](/Users/armand/Development/RingRift/ai-service/app/coordination/event_router.py) - Router integration
- [`event_mappings.py`](/Users/armand/Development/RingRift/ai-service/app/coordination/event_mappings.py) - Event type mappings
- [`data_events.py`](/Users/armand/Development/RingRift/ai-service/app/distributed/data_events.py) - DataEventType enum
- [`stage_events.py`](/Users/armand/Development/RingRift/ai-service/app/coordination/stage_events.py) - StageEvent enum
