# ADR-014: Event System Consolidation

**Status**: Proposed
**Date**: 2025-12-29
**Author**: Claude (AI-assisted architecture review)

## Context

The RingRift AI service has evolved to include three separate event bus systems:

1. **`event_router.py`** - Primary in-memory event bus
   - `DataEventType` enum with 211 event types
   - `get_event_bus()` / `get_router()` accessors
   - Supports async handlers with fire-and-forget

2. **`stage_events.py`** - Pipeline stage events
   - `StageEvent` dataclass for pipeline transitions
   - `get_stage_event_bus()` accessor
   - Used by `DataPipelineOrchestrator`

3. **`cross_process_events.py`** - SQLite-backed cross-process events
   - Persisted to SQLite for crash recovery
   - `get_cross_process_queue()` accessor
   - Used for events that must survive process restarts

### Problems

1. **Consumer Confusion**: New developers are unsure which bus to subscribe to
2. **Duplicate Subscriptions**: Some handlers subscribe to multiple buses for same events
3. **Event Mapping Complexity**: `DATA_TO_CROSS_PROCESS_MAP` and `STAGE_TO_DATA_EVENT_MAP` add indirection
4. **Inconsistent Emission**: Some events emitted on wrong bus, missing subscribers

### Current Architecture

```
                                    ┌─────────────────────┐
                                    │   event_router.py   │
                                    │  DataEventType(207) │
                                    │  Primary in-memory  │
                                    └─────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
                    ▼                         ▼                         ▼
          ┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
          │ stage_events.py │      │ cross_process.py │      │ DataPipeline     │
          │  StageEvent     │      │  SQLite-backed   │      │ FeedbackLoop     │
          │  Pipeline stages│      │  Crash-safe      │      │ SelfplayScheduler│
          └─────────────────┘      └──────────────────┘      └──────────────────┘
```

## Decision

Consolidate to a single unified event type system while preserving transport flexibility:

### Phase 1: Unified Event Type Enum (Q1 2026)

1. **Merge all event types into `DataEventType`**
   - Add missing stage events as `STAGE_*` prefixed types
   - Deprecate `StageEvent` dataclass in favor of `DataEvent`

2. **Create single subscription point**
   - `subscribe(event_type)` automatically routes to correct transport
   - Internal bridge handles cross-process persistence when needed

### Phase 2: Transport Abstraction (Q2 2026)

1. **EventTransport Protocol**

   ```python
   @runtime_checkable
   class EventTransport(Protocol):
       async def publish(self, event: DataEvent) -> None: ...
       async def subscribe(self, event_type: DataEventType, handler: Callable) -> None: ...
   ```

2. **Transport implementations**:
   - `InMemoryTransport` - Current event_router behavior (default)
   - `CrossProcessTransport` - SQLite-backed persistence
   - `HybridTransport` - Routes based on event type configuration

3. **Single `get_router()` entry point**
   ```python
   router = get_router()
   router.subscribe(DataEventType.TRAINING_COMPLETED, handler)
   router.publish(event)  # Transport selected automatically
   ```

### Phase 3: Deprecation (Q3 2026)

1. Deprecate direct imports from `stage_events.py` and `cross_process_events.py`
2. Add deprecation warnings with migration guidance
3. Archive deprecated modules in Q4 2026

## Consequences

### Positive

- **Simplified Mental Model**: One subscription point, one event type enum
- **Reduced Bugs**: No more events emitted on wrong bus
- **Easier Onboarding**: Clear guidance for new contributors
- **Better Testing**: Single mock point for event tests

### Negative

- **Migration Effort**: ~40 files need import updates
- **Backward Compatibility**: Need shim modules during transition
- **Transport Complexity**: Internal routing logic more complex

### Neutral

- Cross-process persistence behavior unchanged (just accessed differently)
- Event handler signatures remain the same

## Migration Path

### For Event Publishers

```python
# Before (ambiguous)
from app.coordination.stage_events import emit_stage_event
from app.coordination.event_router import emit_training_complete

# After (unified)
from app.coordination.event_router import publish, DataEventType, DataEvent

event = DataEvent(
    event_type=DataEventType.TRAINING_COMPLETED,
    payload={"config_key": "hex8_2p"},
)
await publish(event)
```

### For Event Subscribers

```python
# Before (multiple buses)
bus = get_event_bus()
stage_bus = get_stage_event_bus()
bus.subscribe(DataEventType.TRAINING_COMPLETED, handler)
stage_bus.subscribe("pipeline_export", handler2)

# After (unified)
router = get_router()
router.subscribe(DataEventType.TRAINING_COMPLETED, handler)
router.subscribe(DataEventType.STAGE_EXPORT_STARTED, handler2)
```

## Files Affected

### Core Changes

- `app/coordination/event_router.py` - Add transport abstraction
- `app/coordination/stage_events.py` - Deprecate, add shim
- `app/coordination/cross_process_events.py` - Deprecate, add shim
- `app/coordination/event_mappings.py` - Remove after migration

### Consumer Updates (~40 files)

- `app/coordination/data_pipeline_orchestrator.py`
- `app/coordination/feedback_loop_controller.py`
- `app/coordination/selfplay_scheduler.py`
- `scripts/p2p_orchestrator.py`
- And ~36 more coordination modules

## Timeline

| Phase   | Target  | Deliverables                             |
| ------- | ------- | ---------------------------------------- |
| Phase 1 | Q1 2026 | Unified enum, single subscription point  |
| Phase 2 | Q2 2026 | Transport abstraction, automatic routing |
| Phase 3 | Q3 2026 | Deprecation warnings, migration complete |
| Cleanup | Q4 2026 | Archive deprecated modules               |

## References

- ADR-001: Event-Driven Architecture (original design)
- `docs/EVENT_SYSTEM_REFERENCE.md`: Current event catalog
- `app/coordination/event_router.py`: Primary implementation
