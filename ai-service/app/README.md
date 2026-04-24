# AI Service Python Package Guide

`app/` is the active Python package for RingRift. It contains the inference
service, the replay/parity mirror of the TypeScript engine, the training stack,
and a large amount of operational coordination code.

If you are new to this tree, do not read it alphabetically. Start with the
active boundaries and treat compatibility facades as secondary.

## Start Here

1. [`main.py`](/ai-service/app/main.py) for the FastAPI service boundary.
2. [`models`](/ai-service/app/models) for the canonical Python game and AI config types.
3. [`game_engine`](/ai-service/app/game_engine) for the Python replay engine that mirrors the TypeScript rules.
4. [`training`](/ai-service/app/training) for the supported training stack and minimal-loop-adjacent helpers.
5. [`db`](/ai-service/app/db) for replay databases and recording utilities.

## Directory Map

- [`ai`](/ai-service/app/ai): AI implementations, model loaders, legacy compatibility shims, and search code.
- [`coordination`](/ai-service/app/coordination): cluster/event/orchestration layer. Large, historically layered, and not the best first read.
- [`distributed`](/ai-service/app/distributed): cross-node sync, manifests, circuit breakers, and shared distribution helpers.
- [`events`](/ai-service/app/events): unified event taxonomy and compatibility aliases.
- [`training`](/ai-service/app/training): data loading, model lifecycle, Elo services, orchestration, and trainer-side utilities.
- [`rules`](/ai-service/app/rules): canonical move/phase contracts, validators, and replay-history invariants.
- [`metrics`](/ai-service/app/metrics), [`monitoring`](/ai-service/app/monitoring), [`observability`](/ai-service/app/observability): instrumentation and health reporting surfaces.
- [`providers`](/ai-service/app/providers): cloud and host-provider abstractions.
- [`utils`](/ai-service/app/utils): shared low-level helpers.

## Active vs Legacy

Active code should import from `app.*` only.

- Compatibility facades still exist in places like [`app.ai`](/ai-service/app/ai) and [`app.training`](/ai-service/app/training), but they are being narrowed deliberately.
- Archived code lives under [`archive/`](/ai-service/archive) and should not be imported by new modules.
- When you see a private `_deprecated_*` module under `app/`, that is a deliberate compatibility shim kept in the active tree so `app/` does not depend on `archive/`.

## Best Reading Order By Goal

- For inference/service work: `main.py` -> `ai/` -> `models/`.
- For parity/debugging: `models/` -> `game_engine/` -> `rules/` -> `db/`.
- For training: `training/` -> `db/` -> `rules/` -> selected `coordination/` helpers.
- For operations: `scripts/README.md` first, then `coordination/` and `distributed/`.

## Files To Treat Carefully

- [`coordination/__init__.py`](/ai-service/app/coordination/__init__.py), [`training/__init__.py`](/ai-service/app/training/__init__.py), and other package facades are compatibility boundaries, not ideal examples of new module design.
- [`events/__init__.py`](/ai-service/app/events/__init__.py) and [`events/types.py`](/ai-service/app/events/types.py) expose deprecated aliases for compatibility; runtime stage-event code should prefer [`coordination/stage_events.py`](/ai-service/app/coordination/stage_events.py).
- [`archive/`](/ai-service/archive) is historical reference, not part of the supported active tree.

## Related Guides

- [`../README.md`](/ai-service/README.md) for the AI service overview.
- [`../scripts/README.md`](/ai-service/scripts/README.md) for the supported operational scripts.
- [`../../docs/REPRODUCIBILITY.md`](/docs/REPRODUCIBILITY.md) and [`../../docs/data/results_evidence_manifest.json`](/docs/data/results_evidence_manifest.json) for checked-in data and result provenance.
