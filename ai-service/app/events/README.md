# RingRift Event System

**Single source of truth for all event types used across the RingRift AI service.**

## Overview

This package consolidates 3 previously separate event type definitions into a unified `RingRiftEventType` enum with **172 event types** organized into **15 categories**.

### Previous State (Duplicated)

- `app.distributed.data_events.DataEventType` (~140 types)
- `app.coordination.stage_events.StageEvent` (22 types)
- Cross-process event patterns (scattered)

### Current State (Unified)

- `app.events.types.RingRiftEventType` (172 types, single source of truth)
- Full backwards compatibility maintained
- Category-based organization
- Comprehensive documentation

## Quick Start

```python
from app.events import RingRiftEventType, EventCategory

# Use unified event type
event_type = RingRiftEventType.TRAINING_COMPLETED

# Check category
category = EventCategory.from_event(event_type)
assert category == EventCategory.TRAINING
```

## Event Categories

The 172 event types are organized into 15 categories:

| Category         | Count | Description                      |
| ---------------- | ----- | -------------------------------- |
| **data**         | 13    | Data collection, sync, freshness |
| **training**     | 10    | Training lifecycle events        |
| **evaluation**   | 8     | Model evaluation and Elo         |
| **promotion**    | 9     | Model promotion workflow         |
| **curriculum**   | 4     | Curriculum management            |
| **selfplay**     | 3     | Selfplay operations              |
| **optimization** | 14    | CMA-ES, NAS, PBT                 |
| **quality**      | 14    | Data quality monitoring          |
| **regression**   | 6     | Regression detection             |
| **cluster**      | 16    | Cluster/P2P operations           |
| **system**       | 31    | Daemons, health, resources       |
| **work**         | 9     | Work queue events                |
| **stage**        | 20    | Pipeline stage completions       |
| **sync**         | 6     | Synchronization and locking      |
| **task**         | 9     | Task lifecycle                   |

## Usage Examples

### Basic Usage

```python
from app.events import RingRiftEventType

# Reference an event type
if event.type == RingRiftEventType.MODEL_PROMOTED:
    handle_promotion(event)
```

### Get Events by Category

```python
from app.events import EventCategory, get_events_by_category

# Get all training-related events
training_events = get_events_by_category(EventCategory.TRAINING)
# Returns: [TRAINING_STARTED, TRAINING_PROGRESS, TRAINING_COMPLETED, ...]
```

### Check Cross-Process Events

```python
from app.events import is_cross_process_event, RingRiftEventType

# Check if event should be propagated across processes
if is_cross_process_event(RingRiftEventType.MODEL_PROMOTED):
    bridge_to_cross_process_queue(event)
```

### Backwards Compatibility

Existing code using old imports continues to work:

```python
# Old code (still works)
from app.distributed.data_events import DataEventType
event = DataEventType.TRAINING_COMPLETED

# New code (recommended)
from app.events import RingRiftEventType
event = RingRiftEventType.TRAINING_COMPLETED
```

## Files

- **`types.py`**: Main module with `RingRiftEventType` enum and utilities
- **`__init__.py`**: Package exports for clean imports

## Migration Guide

### For New Code

✅ Use `app.events.RingRiftEventType` directly
✅ Use `EventCategory` for event grouping
✅ Import from `app.events` package

### For Existing Code

✅ No changes required (backwards compatible)
⚠️ Consider migrating imports to `app.events` when convenient
⚠️ Old imports (`data_events`, `stage_events`) may show deprecation warnings

### Migration Steps (Optional)

1. Update imports:

   ```python
   # Before
   from app.distributed.data_events import DataEventType

   # After
   from app.events import RingRiftEventType
   ```

2. Update event references:

   ```python
   # Before
   if event_type == DataEventType.TRAINING_COMPLETED:

   # After
   if event_type == RingRiftEventType.TRAINING_COMPLETED:
   ```

3. For stage events:

   ```python
   # Before
   from app.coordination.stage_events import StageEvent
   event = StageEvent.TRAINING_COMPLETE

   # After
   from app.events import RingRiftEventType
   event = RingRiftEventType.STAGE_TRAINING_COMPLETE
   ```

## Benefits

1. **Single Source of Truth**: All events defined in one place
2. **Better Organization**: Events grouped by category
3. **Type Safety**: Single enum ensures consistency
4. **Discoverability**: See all events in one file
5. **Maintainability**: Add new events in one location
6. **Backwards Compatible**: Existing code works unchanged

## Cross-Process Events

32 event types are flagged for cross-process propagation. These include:

- **Success events**: MODEL_PROMOTED, TRAINING_COMPLETED, etc.
- **Failure events**: TRAINING_FAILED, PROMOTION_FAILED, etc.
- **Cluster events**: HOST_ONLINE, HOST_OFFLINE, etc.
- **Trigger events**: CMAES_TRIGGERED, NAS_TRIGGERED, etc.
- **Regression events**: REGRESSION_CRITICAL, etc.

Access via:

```python
from app.events import CROSS_PROCESS_EVENT_TYPES, is_cross_process_event
```

## Documentation

Each event type has a comprehensive docstring:

```python
TRAINING_COMPLETED = "training_completed"
"""Training completed successfully."""

MODEL_PROMOTED = "model_promoted"
"""Model successfully promoted to production."""
```

## See Also

- `app/distributed/data_events.py` - Original data event definitions (deprecated)
- `app/coordination/stage_events.py` - Original stage event definitions (deprecated)
- `app/coordination/event_router.py` - Unified event routing system
