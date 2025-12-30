# ADR-001: Event-Driven Architecture for Training Pipeline

**Status**: Accepted
**Date**: December 2025
**Author**: RingRift AI Team

> **Note:** This ADR reflects Dec 2025 counts. Current `DataEventType` has 211 events; see `ai-service/docs/EVENT_SYSTEM_REFERENCE.md`.

## Context

The RingRift AI training pipeline consists of multiple stages (selfplay, data sync, export, training, evaluation, promotion) that need to be coordinated across a distributed cluster of 20+ GPU nodes.

Initially, the pipeline used polling and direct method calls, which led to:

- Tight coupling between components
- Difficulty adding new pipeline stages
- No visibility into pipeline progress
- Race conditions and deadlocks

## Decision

Adopt an **event-driven architecture** using a unified event bus with the following design:

### Event Bus Design

1. **DataEventType enum** (`app/distributed/data_events.py`): 151 event types covering all pipeline stages (Dec 2025 snapshot)
2. **EventRouter** (`app/coordination/event_router.py`): Unified router bridging three underlying buses:
   - `DataEventBus`: In-memory async event bus
   - `StageEventBus`: Pipeline stage transitions
   - `CrossProcessQueue`: P2P cluster-wide events

### Event Flow

```
SELFPLAY_COMPLETE → DATA_SYNC_TRIGGERED → NPZ_EXPORT_COMPLETE →
TRAINING_STARTED → TRAINING_COMPLETED → EVALUATION_COMPLETE →
MODEL_PROMOTED → MODEL_DISTRIBUTED
```

### Event Naming Convention

- `*_STARTED`: Action beginning
- `*_COMPLETE` / `*_COMPLETED`: Successful completion
- `*_FAILED`: Failure with error info
- `*_UPDATED`: State change (e.g., `ELO_UPDATED`)

## Consequences

### Positive

- Loose coupling: Components only depend on event types, not other components
- Extensibility: New subscribers can be added without modifying emitters
- Visibility: Event logging provides complete pipeline trace
- Reliability: Failed handlers don't block other handlers

### Negative

- Debugging complexity: Event chains can be harder to trace than direct calls
- Latency: Event dispatch adds small overhead (~1ms per event)
- Memory: Event history for deduplication requires bounded storage

## Implementation Notes

- All events include `source`, `timestamp`, and unique `event_id`
- SHA256 content-based deduplication prevents duplicate processing
- Fire-and-forget semantics with error callbacks for async handlers
- Event subscribers should be idempotent (handle duplicate events gracefully)

## Related ADRs

- ADR-002: Daemon Lifecycle Management (uses events for daemon coordination)
- ADR-003: PFSP Opponent Selection (subscribes to MODEL_PROMOTED events)
