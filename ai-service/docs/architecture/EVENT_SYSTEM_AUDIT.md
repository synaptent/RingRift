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
- `tests/contracts/test_event_system_canonical.py`

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

## Current Backlog

The active P2P runtime path has now been migrated onto `event_emission_helpers`.
This includes the orchestrator, partition healing, voter/relay health, loop base
helpers, leader probe/maintenance, training sync, tournament data pipeline, peer
network bootstrap, and the cluster training executor.

The remaining `emit_event` references in the repository are intentionally narrow:

- `app/coordination/event_router_compat_emitters.py`
  Compatibility shim that preserves old call sites while delegating to the router.
- `app/coordination/event_fallback_queue.py`
  Documentation/examples around fallback queue usage, plus `publish_sync` usage for
  its own sync-complete event.
- `app/coordination/event_emission_helpers.py`
  Historical explanatory text that refers to the old migration target.
- `app/core/async_context.py`
  Generic async callback example unrelated to the coordination event router.
- P2P loop docstrings and examples such as `scripts/p2p/models.py`
  Non-runtime examples that mention `emit_event`.

`tests/contracts/test_event_system_canonical.py` now enforces the active-code
boundary with an AST scan, so runtime `emit_event` regressions in
`app/coordination` or `scripts/p2p` should fail the contract gate immediately.

## Guardrails For New Work

New coordination daemon code should:

- Import `safe_emit_event` or `safe_emit_event_async` from `app.coordination.event_emission_helpers`.
- Use `SafeEventEmitterMixin` only when a class-level `_event_source` wrapper is useful.
- Avoid direct `event_router.emit_event` imports outside event infrastructure modules.
- Add a subscription matrix assertion when a profile starts a daemon that emits a pipeline-critical event.
- Keep event payload schemas aligned with `EVENT_PAYLOAD_SCHEMAS.md` when adding or changing payload fields.

The event-system cleanup is therefore complete for the active coordination and
P2P runtime path. What remains is compatibility/documentation debt, not
supported-path runtime debt, and the new contract test keeps that boundary
explicit.
