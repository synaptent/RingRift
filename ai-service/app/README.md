# AI Service Python Package Guide

`app/` is the active Python package for RingRift. It contains the inference
service, the replay/parity mirror of the TypeScript engine, the training stack,
and a large amount of operational coordination code.

If you are new to this tree, do not read it alphabetically. Start with the
active boundaries and treat compatibility facades as secondary.

## Start Here

1. [`main.py`](/Users/armand/Development/RingRift/ai-service/app/main.py) for the FastAPI service boundary.
2. [`models`](/Users/armand/Development/RingRift/ai-service/app/models) for the canonical Python game and AI config types.
3. [`game_engine`](/Users/armand/Development/RingRift/ai-service/app/game_engine) for the Python replay engine that mirrors the TypeScript rules.
4. [`training`](/Users/armand/Development/RingRift/ai-service/app/training) for the supported training stack and minimal-loop-adjacent helpers.
5. [`db`](/Users/armand/Development/RingRift/ai-service/app/db) for replay databases and recording utilities.

## Directory Map

- [`ai`](/Users/armand/Development/RingRift/ai-service/app/ai): AI implementations, model loaders, legacy compatibility shims, and search code.
- [`coordination`](/Users/armand/Development/RingRift/ai-service/app/coordination): cluster/event/orchestration layer. Large, historically layered, and not the best first read.
- [`distributed`](/Users/armand/Development/RingRift/ai-service/app/distributed): cross-node sync, manifests, circuit breakers, and shared distribution helpers.
- [`events`](/Users/armand/Development/RingRift/ai-service/app/events): unified event taxonomy and compatibility aliases.
- [`training`](/Users/armand/Development/RingRift/ai-service/app/training): data loading, model lifecycle, Elo services, orchestration, and trainer-side utilities.
- [`rules`](/Users/armand/Development/RingRift/ai-service/app/rules): canonical move/phase contracts, validators, and replay-history invariants.
- [`metrics`](/Users/armand/Development/RingRift/ai-service/app/metrics), [`monitoring`](/Users/armand/Development/RingRift/ai-service/app/monitoring), [`observability`](/Users/armand/Development/RingRift/ai-service/app/observability): instrumentation and health reporting surfaces.
- [`providers`](/Users/armand/Development/RingRift/ai-service/app/providers): cloud and host-provider abstractions.
- [`utils`](/Users/armand/Development/RingRift/ai-service/app/utils): shared low-level helpers.

## Active vs Legacy

Active code should import from `app.*` only.

- Compatibility facades still exist in places like [`app.ai`](/Users/armand/Development/RingRift/ai-service/app/ai) and [`app.training`](/Users/armand/Development/RingRift/ai-service/app/training), but they are being narrowed deliberately.
- Archived code lives under [`archive/`](/Users/armand/Development/RingRift/ai-service/archive) and should not be imported by new modules.
- When you see a private `_deprecated_*` module under `app/`, that is a deliberate compatibility shim kept in the active tree so `app/` does not depend on `archive/`.

## Best Reading Order By Goal

- For inference/service work: `main.py` -> `ai/` -> `models/`.
- For parity/debugging: `models/` -> `game_engine/` -> `rules/` -> `db/`.
- For training: `training/` -> `db/` -> `rules/` -> selected `coordination/` helpers.
- For operations: `scripts/README.md` first, then `coordination/` and `distributed/`.

## Files To Treat Carefully

- [`coordination/__init__.py`](/Users/armand/Development/RingRift/ai-service/app/coordination/__init__.py), [`training/__init__.py`](/Users/armand/Development/RingRift/ai-service/app/training/__init__.py), and other package facades are compatibility boundaries, not ideal examples of new module design.
- [`events/__init__.py`](/Users/armand/Development/RingRift/ai-service/app/events/__init__.py) and [`events/types.py`](/Users/armand/Development/RingRift/ai-service/app/events/types.py) expose deprecated aliases for compatibility; runtime stage-event code should prefer [`coordination/stage_events.py`](/Users/armand/Development/RingRift/ai-service/app/coordination/stage_events.py).
- [`archive/`](/Users/armand/Development/RingRift/ai-service/archive) is historical reference, not part of the supported active tree.

## Related Guides

- [`../README.md`](/Users/armand/Development/RingRift/ai-service/README.md) for the AI service overview.
- [`../scripts/README.md`](/Users/armand/Development/RingRift/ai-service/scripts/README.md) for the supported operational scripts.
- [`../TRAINING_DATA_REGISTRY.md`](/Users/armand/Development/RingRift/ai-service/TRAINING_DATA_REGISTRY.md) for canonical vs legacy data provenance.
