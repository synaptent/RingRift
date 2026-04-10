# Event System Audit

**Date:** April 10, 2026
**Scope:** Coordination daemons, master loop, and P2P infrastructure paths that may be reused alongside the supported minimal training loop.

## Current Decision

The canonical event emission path for active coordination code is:

```python
from app.coordination.event_emission_helpers import safe_emit_event
from app.coordination.event_emission_helpers import safe_emit_event_async
```

`app.coordination.safe_event_emitter.SafeEventEmitterMixin` is still supported for classes that expose `_safe_emit_event()` or `_safe_emit_event_async()`, but it is a compatibility layer. It delegates to `event_emission_helpers`, and tests now enforce that it does not bypass the consolidated helper.

The router remains the delivery layer. New daemon code should not import `event_router.emit_event` directly.

## What Was Audited

The cleanup focused on four contracts:

- Emitters in the active coordination path should use `safe_emit_event` or `safe_emit_event_async`.
- Event import paths should not drift back to direct `event_router.emit_event` imports in active coordination modules.
- Dead-event diagnostics should surface emitted event types that have no subscribers.
- The `lean` master-loop profile should include at least one subscriber daemon for the pipeline-critical events it can trigger.

The current automated coverage is:

- `tests/unit/coordination/test_event_subscription_completeness.py`
- `tests/unit/coordination/test_safe_event_emitter.py`
- `tests/unit/coordination/test_infrastructure_quality_contracts.py`

## Findings

The active coordination path now has a safer emission contract. `safe_event_emitter.safe_emit_event()` is tested as a compatibility wrapper over `event_emission_helpers.safe_emit_event()`, and the import-path audit covers active `app/coordination` modules plus `scripts/master_loop.py`.

The lean profile event matrix is explicitly tested for these pipeline-critical events:

| Event                  | Required coverage intent                                      |
| ---------------------- | ------------------------------------------------------------- |
| `SELFPLAY_COMPLETE`    | At least one lean daemon receives selfplay completion signals |
| `DATA_SYNC_COMPLETED`  | Data pipeline receives sync completion signals                |
| `TRAINING_COMPLETED`   | Feedback/data pipeline can react to training completion       |
| `EVALUATION_COMPLETED` | Feedback/auto-promotion can react to evaluation completion    |
| `MODEL_PROMOTED`       | Feedback/model distribution/selfplay can react to promotions  |

Dead-event detection is covered through `UnifiedEventRouter.get_orphaned_events()` and `get_status()["orphan_events"]`, so future regressions should show up as emitted event types without subscribers.

## Known Remaining Backlog

There are still legacy or lower-level compatibility paths with direct `emit_event` references, especially under `scripts/p2p/**`. A source scan after the cleanup found 23 files under `app/coordination` and `scripts/p2p` with direct `emit_event` references or fallback examples, excluding `event_router.py` itself. Some are event infrastructure or fallback code such as `event_emission_helpers.py` and `event_fallback_queue.py`. The remaining P2P references are not automatically safe to rewrite without behavioral review because several sit inside leadership, partition-healing, health, and loop compatibility code.

Important examples to audit before promoting more P2P code into the supported path:

- `scripts/p2p/voter_health_monitor.py`
- `scripts/p2p/relay_leader_propagator.py`
- `scripts/p2p/partition_healer.py`
- `scripts/p2p/loops/base.py`
- `scripts/p2p/loops/leader_probe_loop.py`
- `scripts/p2p/orchestrators/peer_network_orchestrator.py`
- `scripts/p2p/work_executors/training_executor.py`

These should be migrated deliberately to `event_emission_helpers` or to a P2P-local facade that delegates to the consolidated helper, with focused tests for the surrounding recovery behavior.

## Guardrails For New Work

New coordination daemon code should:

- Import `safe_emit_event` or `safe_emit_event_async` from `app.coordination.event_emission_helpers`.
- Use `SafeEventEmitterMixin` only when a class-level `_event_source` wrapper is useful.
- Avoid direct `event_router.emit_event` imports outside event infrastructure modules.
- Add a subscription matrix assertion when a profile starts a daemon that emits a pipeline-critical event.
- Keep event payload schemas aligned with `EVENT_PAYLOAD_SCHEMAS.md` when adding or changing payload fields.

The event-system cleanup is therefore not "done" for all legacy P2P code. It is done for the active coordination path covered by the current tests, and the remaining P2P backlog is documented for incremental reuse.
