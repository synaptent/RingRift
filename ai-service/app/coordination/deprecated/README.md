# Deprecated Coordination Package

This package is no longer a live compatibility surface for day-to-day coordination code.

As of April 2026 it serves two narrow purposes:

- `app.coordination.deprecated` exposes archived coordination module names through `dir()`
  and raises clear `ImportError` messages that point callers at their canonical replacements.
- `_deprecated_sync_coordinator.py` remains as the one still-tested legacy scheduling module
  while its final callers are drained.

New code should import from the canonical coordination modules directly.

## Current Package Behavior

The package root does not lazily proxy old implementations anymore. Instead,
`app.coordination.deprecated.<name>` intentionally fails with a migration error
for the archived names listed below.

Archived names exposed by the package root:

| Archived Name               | Use Instead                                                          |
| --------------------------- | -------------------------------------------------------------------- |
| `cross_process_events`      | `app.coordination.event_router`                                      |
| `event_emitters`            | `app.coordination.event_router`                                      |
| `health_check_orchestrator` | `app.coordination.unified_health_manager`                            |
| `host_health_policy`        | `app.coordination.unified_health_manager`                            |
| `system_health_monitor`     | `app.coordination.unified_health_manager`                            |
| `auto_evaluation_daemon`    | `app.coordination.daemon_manager` with `EVALUATION_DAEMON`           |
| `sync_coordinator`          | `app.coordination.auto_sync_daemon` + `app.coordination.sync_router` |
| `queue_populator_daemon`    | `app.coordination.unified_queue_populator`                           |

Example:

```python
import app.coordination.deprecated as deprecated

"sync_coordinator" in dir(deprecated)

try:
    deprecated.sync_coordinator
except ImportError as exc:
    print(exc)
```

## Remaining Legacy Module

The one remaining implementation in this directory is:

- `app.coordination.deprecated._deprecated_sync_coordinator`

That module still exists because its legacy scheduling behavior remains under
focused tests. It is not the preferred import path for new code.

For current sync coordination:

- Scheduling and policy: `app.coordination.auto_sync_daemon`
- Routing and orchestration: `app.coordination.sync_router`
- Distributed execution: `app.distributed.sync_coordinator`

## Historical Notes

Earlier consolidation docs described a larger package layout under
`app.coordination.core`, `app.coordination.cluster`, and similar migration-era
groupings. That documentation is historical now and should not be treated as
the current public API for this package.

For archived consolidation notes, use the checked-in historical docs under:

- `ai-service/docs/archive/coordination_consolidation_2025_12/`

## Guidance

- Do not add new imports from `app.coordination.deprecated`.
- If a caller still needs one of the archived names, move it to the canonical
  module named in the migration table above.
- If you are working on legacy sync scheduling behavior, touch
  `_deprecated_sync_coordinator.py` directly and keep the focused coordination
  tests green.
