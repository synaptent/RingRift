# Codebase Quality Program

This document is the durable execution plan for raising the RingRift codebase toward a high standard of quality, presentability, understandability, and maintainability without destabilizing the supported training path.

Status is current as of April 13, 2026.

## Baseline Scores

These are the current working scores for the overall repository:

| Dimension         | Current | Target |
| ----------------- | ------: | -----: |
| Code quality      |   `7.5` |  `9.0` |
| Presentability    |   `7.0` |  `9.0` |
| Understandability |   `5.5` |  `8.5` |
| Maintainability   |   `6.0` |  `8.5` |

The goal is not to cosmetically relabel the repo. The goal is to make the supported path, the public APIs, and the active operational surface easier to trust and easier to change safely.

## Guardrails

- Do not restart or churn infrastructure unless a code change directly requires it.
- Prefer source-of-truth reduction over adding more wrappers.
- Prefer deleting dead surfaces over maintaining historical compatibility indefinitely.
- Use small, verified commits.
- Keep claims tied to current checked-in evidence, not oral history.

## Workstreams

### 1. Source-of-truth reduction

Objective: reduce contradictory docs, stale snapshots, and duplicated project-state narratives.

Deliverables:

- one authoritative results narrative
- one authoritative machine-readable results snapshot
- one authoritative fleet/runtime description
- explicit labeling of point-in-time operational memos vs current state

Acceptance criteria:

- `docs/RESULTS.md`, `docs/RESEARCH_SNAPSHOT.md`, `docs/PROJECT_BRIEF.md`, and `docs/data/results_snapshot.json` agree on headline numbers
- stale operational notes are clearly marked as historical snapshots

### 2. Public API contraction

Objective: make package entrypoints explicit and small.

Deliverables:

- reduced lazy-export and compatibility-shim surface
- package `__init__.py` files that expose only intentional public APIs
- fewer deprecated import paths

Acceptance criteria:

- public package APIs are discoverable from package `__init__.py`
- deleted shims are replaced with canonical import paths and passing package tests

### 3. Coordination decomposition

Objective: reduce the maintenance cost of `ai-service/app/coordination`.

Deliverables:

- dead-code deletion
- removal of stale re-export modules
- narrower module boundaries by responsibility
- smaller package-level cognitive load

Acceptance criteria:

- coordination public surface is materially smaller
- internal modules no longer depend on obsolete compatibility paths
- package tests and import smoke tests stay green

### 4. Quality-gate hardening

Objective: make drift visible quickly.

Deliverables:

- warning budget of `0`
- tracked skip budget with named reasons
- import/deprecation smoke tests for public APIs
- stale-doc / stale-snapshot checks
- package API tests for active packages

Acceptance criteria:

- warning count stays at `0`
- optional skips are collapsed and intentional
- drift between docs and checked-in snapshots is caught automatically

### 5. Navigability and onboarding

Objective: make the repo easier to understand without tribal knowledge.

Deliverables:

- clear architecture entrypoints
- curated script inventory
- subsystem maps for the active surfaces
- tighter index and repository-map guidance

Acceptance criteria:

- a new engineer can find the supported path, current results, and active APIs quickly
- historical or archival areas are clearly separated from active code

## Active Batch Order

1. Source-of-truth reduction for results, snapshots, and public summaries.
2. Continue package API contraction in `ai-service/app/coordination`.
3. Add automated drift guards for docs, warnings, skips, and public APIs.
4. Continue subsystem-by-subsystem dead-code deletion and simplification.

## Current Batch

Status: in progress

Current objectives:

- keep shrinking and locking the remaining `app.coordination` public surface one submodule at a time
- prefer narrow package-surface ratchets and explicit `dir()` discoverability before larger refactors
- protect the minimal training loop path with direct smoke/import-hygiene guards before assuming package-surface refactors are safe for trainer deployment
- keep each batch small enough to verify with focused coordination tests and clean checkpoints

## Latest Progress

- Public docs and results snapshots were aligned to the April 13 state and guarded by consistency tests.
- Runtime coordination imports were reduced from three intentional top-level facade consumers to zero real runtime facade consumers outside the package itself.
- `run_random_selfplay.py`, the CLI coordination-status command, and `scripts/p2p/startup_infrastructure.py` now import explicit owning modules instead of the top-level `app.coordination` package.
- The facade-shrink phase has started: coordination status and aggregated health reporting now live in `app.coordination.status_reporting`, with `app.coordination.__init__` reduced to compatibility wrappers for those helpers.
- Coordination bootstrap, shutdown, and heartbeat helpers now live in `app.coordination.lifecycle`, further reducing `app.coordination.__init__` from a logic owner to a compatibility entrypoint.
- The same contraction pattern is now started for `app.training`: runtime consumers were moved off the top-level training facade and a new import-hygiene ratchet confirms zero real runtime `from app.training import ...` consumers outside the package.
- The next contraction seam is `app.training.__init__`: it is still the largest remaining package facade, so the first step is to lock its declared surface under tests and make `dir(app.training)` reflect that public API intentionally.
- `app.training` now has that explicit package-surface ratchet too: focused tests cover key exports, legacy compatibility entries, and `dir(app.training)` discoverability.
- The same runtime-facade ratchet now covers `app.distributed` too: CMA-ES and archival distributed training scripts now import owning submodules directly, and the distributed import-hygiene ratchet confirms zero `app.distributed` facade imports outside the package.
- `app.distributed` now also has an explicit package-surface ratchet: focused tests cover key public exports, lazy deprecated symbols, and `dir()` discoverability.
- `app.metrics` now has the first half of the same treatment too: runtime facade consumers were drained, and the next acceptance bar is a package-surface ratchet so `dir(app.metrics)` and its declared exports stay aligned under test.
- `app.metrics` now has that package-surface ratchet too: focused tests cover key exports, rollback helpers, and `dir(app.metrics)` discoverability.
- The next large package seam is `app.errors`: it appears to be an intentional public entrypoint rather than a facade to drain immediately, so the first move there is to lock its declared hierarchy and aliases under package-surface tests before considering any contraction.
- `app.errors` now has that package-surface ratchet too: focused tests lock key exports and ensure `dir(app.errors)` stays aligned with its declared public surface.
- The next narrow coordination seam is `app.coordination.interfaces`: it is already a small, dependency-light protocol module, so the right move there is to ratchet its declared protocol surface and make package discovery intentional instead of leaving it implicit.
- `app.coordination.interfaces` now has that package-surface ratchet too: focused tests lock its protocol exports, and `__dir__()` now makes the intended interface surface explicit for discoverability and future regression checks.
- `app.coordination.queue_strategies` now follows the same pattern too: its tiny mixin package surface is locked under focused tests, and `__dir__()` now exposes that public package surface intentionally instead of relying on implicit module behavior.
- `app.coordination.availability` now follows the same pattern too: its package export list is locked under focused tests, and `__dir__()` now makes the node-monitor / recovery / provisioning surface explicit for discoverability.
- `app.coordination.health` now follows the same pattern too: its canonical health-type package surface is locked under focused tests, and `__dir__()` now makes that public health API explicit for discovery and drift checks.
- `app.coordination.feedback`, `app.coordination.mixins`, and `app.coordination.node_availability` now follow the same pattern too: their package surfaces are locked under focused tests, and each package now exposes its intended public API explicitly via `__dir__()` for discovery and drift checks.
- `app.coordination.providers` now follows the same pattern too: its provider base types, registry exports, and package-level factory functions are locked under focused tests, and `__dir__()` now exposes that public provider API intentionally.
- `app.coordination.hashgraph` now follows the same pattern too: its full consensus / DAG / promotion public surface is locked under focused tests, and `__dir__()` now makes that large but intentional package API explicit for discovery and drift checks.
- `app.coordination.node_availability.providers` now follows the same pattern too: its provider-checker package surface is locked under focused tests, and `__dir__()` now makes those node-availability adapters explicit for discovery and drift checks.
- The remaining coordination entrypoints now have intentional surfaces too: `app.coordination.runners` exposes an explicit factory-only public API, while `app.coordination.deprecated` now makes archived module names discoverable via `dir()` and keeps migration errors under focused test.
- The minimal training loop now has a direct regression guard too: focused tests lock its subprocess entrypoint to `app.training.train` and ratchet the critical loop files away from top-level `app.training`, `app.coordination`, and `app.distributed` facade imports so future cleanups cannot silently route the trainer through those package entrypoints.
- The minimal-loop rollout path is now guarded too: `scripts/deploy_minimal_loops.sh` runs a local preflight against the minimal-loop test slice before restarting trainer nodes, with an explicit `--skip-preflight` escape hatch for emergencies. The deploy-script contract tests lock that behavior under dry-run.
- The operator docs now point to that supported rollout path more clearly too: `ai-service/README.md`, `ai-service/scripts/README.md`, and `docs/architecture/MINIMAL_LOOP_CONTRACT.md` now describe `deploy_minimal_loops.sh`, the preflight expectation, and `progress.json` as the live stage-status file for trainer work directories.
- Broader repo navigation now matches that story better too: `docs/DEVELOPER_GUIDE.md` and `docs/SCRIPT_INVENTORY.md` now call out the supported minimal-loop operator entrypoints explicitly instead of leaving them buried in the larger script/doc surface.
- The remaining overview docs now align with that same operator contract too: `docs/CURRENT_STATUS.md`, `docs/ARCHITECTURE_OVERVIEW.md`, `docs/REPOSITORY_MAP.md`, and `docs/architecture/TRAINING_INFRASTRUCTURE_STRATEGY.md` now all route readers toward `deploy_minimal_loops.sh`, `progress.json`, and `metrics.jsonl`, and the docs consistency tests ratchet those references so the supported operator story does not drift again.
- The next low-risk package ratchets are landing outside coordination too: `app.interfaces`, `app.analysis`, and `app.db` now expose explicit `__dir__()` surfaces, and focused tests lock their declared package exports so those entrypoints stay intentional instead of relying on implicit module discovery.
- The same package-surface cleanup now covers three more small top-level packages too: `app.observability`, `app.storage`, and `app.notation` now expose explicit `__dir__()` surfaces, and focused package-export tests lock those public entrypoints under cheap verification.
- Another narrow package wave is now covered too: `app.events`, `app.sync`, and `app.validation` now expose explicit `__dir__()` surfaces, and focused package-export tests lock those public entrypoints under the same cheap ratchet pattern.
- The same pattern now covers two more service entrypoints too: `app.evaluation` and `app.execution` now expose explicit `__dir__()` surfaces, and focused package-export tests lock those public entrypoints under cheap verification.
- Another tiny package wave is now covered too: `app.training.env_mixins`, `app.quality.validators`, `app.rules.generators`, `app.mcts`, `app.ai.nnue_registry`, and `app.quality.scorers` now expose explicit `__dir__()` surfaces, and focused package-export tests lock those public entrypoints under the same cheap ratchet pattern.
- Another small top-level package wave is now covered too: `app.rules`, `app.providers`, `app.caching`, `app.cli`, and `app.testing` now expose explicit `__dir__()` surfaces, and focused package-export tests lock those public entrypoints under the same cheap ratchet pattern.
- The next heavier AI-facing entrypoints now follow the same pattern too: `app.ai.archive`, `app.ai.evaluators`, `app.ai.harness`, `app.training.enhancements`, and `app.training.export` now expose explicit `__dir__()` surfaces, and focused package-export tests lock those public entrypoints under cheap verification.
- The central service facades now follow the same pattern too: `app.game_engine`, `app.models`, `app.monitoring`, and `app.routes` now expose explicit `__dir__()` surfaces, and focused package-export tests lock those public entrypoints under the same cheap verification style.
- The main AI and quality facades now follow the same pattern too: `app.ai` and `app.quality` now expose explicit `__dir__()` surfaces, and focused package-export tests lock those public entrypoints while tolerating the expected deprecation warnings from legacy AI aliases.
- The remaining medium lazy facades now follow the same pattern too: `app.config`, `app.core`, and `app.integration` now expose explicit `__dir__()` surfaces, and focused package-export tests lock those public entrypoints under the same cheap verification style.
- The next re-export packages now follow the same pattern too: `app.ai.neural_net`, `app.p2p`, `app.tournament`, and `app.distributed.data_events` now expose explicit `__dir__()` surfaces, and focused package-export tests lock their advertised entrypoints without broadening the verification cost.
- The remaining utility-facing surfaces now follow the same pattern too: `app.utils` now advertises only the root helpers it actually resolves, `app.rules.legacy` now has explicit discoverability, and focused tests ratchet both package surfaces under cheap targeted verification.
- The package READMEs are being brought under the same standards too: `app.integration/README.md` now documents the real root facade and submodule escape hatches, `app.utils/README.md` now states the supported root-vs-submodule split explicitly, and focused docs tests lock that guidance so the examples do not drift back to dead imports.
- The package README cleanup is expanding one seam at a time: `app.interfaces/README.md` now uses the current hashing example (`app.core.zobrist.ZobristHash`) instead of the stale `app.ai`/`app.zobrist` wiring, and the docs ratchet now locks that correction too.
- The package README cleanup is continuing through the service facades too: `app.metrics/README.md` now uses real root helpers (`record_evaluation`, `record_pipeline_stage`, `record_pipeline_iteration`) instead of dead names, and `app.monitoring/README.md` now separates the supported root facade from advanced submodule-only tools like predictive alerts and the training dashboard.
- The validation docs are now being held to the same bar too: `app.validation/README.md` now documents the actual root surface (`validate`, `validate_all`, `each_item`, `is_instance`, `is_non_negative`) instead of stale helpers like `each_value`, `each_key`, `pydantic_validator`, and a nonexistent `strict=True` mode.
- The deprecated coordination docs are now being aligned with the real archive state too: `app.coordination.deprecated/README.md` no longer describes nonexistent package layouts or archive paths, and the docs ratchet now locks its current role as an archived-name sentinel plus the one remaining legacy sync module.
- The package README audit is continuing through older service docs too: `app.distributed/README.md` now uses the real host circuit-breaker reset API (`get_host_breaker()`, `reset(target)`, `reset_all()`) instead of nonexistent helpers, and `app.quality/README.md` now points readers at the supported optimized training pipeline entrypoints instead of a dead `app.training.TrainingPipeline` facade.
- The game-engine docs now match the actual package contract too: `app.game_engine/README.md` no longer mislabels the package as deprecated or shows the wrong `PhaseRequirement` fields, and the docs ratchet now locks the supported `app.game_engine` import surface plus the current phase-requirement example.
- The coordination package guides are being brought up to the same bar too: `app.coordination.cluster/README.md` now reflects the lazy `health` / `transport` / `p2p` package layout and current health helpers, while `app.coordination.providers/README.md` now uses the enum-based root provider API and `ProviderRegistry` instead of dead string helpers and nonexistent provider methods.
- The older provider-manager docs are being aligned too: `app.providers/README.md` now inventories `VastManager`, uses the current manager interface (`list_instances`, `get_instance`, `check_health`, `run_ssh_command`), and stops advertising removed SSH-config helpers.
- The larger coordination and training guides now match the current facade/operator boundary too: `app.coordination/README.md` now describes the package root as a lazy compatibility facade, points trainer rollout to `deploy_minimal_loops.sh`, and uses the current focused coordination test slices instead of dead `mutants/tests` paths, while `app.training/README.md` now distinguishes the local `run_training_loop.py` utility from the supported minimal-loop rollout path and points operators at `minimal_alphazero_loop.py`, `progress.json`, and `metrics.jsonl`.
- The older architecture guides are now being pulled forward too: `app.coordination/EXPORT_TIERS.md` now reflects the current thin lazy facade (`594` exports, `251` LOC, `_exports_*.py`, and focused ratchets) instead of the old mega-facade roadmap, while `app.training/ORCHESTRATOR_GUIDE.md` and the remaining deprecated-module notes now point to the archived `TrainingOrchestrator` compatibility path plus the supported minimal-loop rollout boundary instead of dead `app.training.orchestrated_training` imports.
- The remaining coordination guide drift is being tightened too: `app.coordination/COORDINATOR_GUIDE.md` now uses the archived `TrainingOrchestrator` path and the preferred `@singleton` guidance instead of steering readers toward dead training imports or legacy singleton defaults, while `app.coordination/training/README.md` now matches the current package exports (`get_training_coordinator`, `get_unified_scheduler`, slot helpers) and the canonical `TRAINING_COMPLETED` event name.
- The remaining deprecation reference is now aligned too: `app.coordination/DEPRECATION_GUIDE.md` no longer claims `sync_coordinator.py` is a same-file alias, no longer shows the nonexistent `EventRouter.get_instance()` flow or a future `core/` / `resources/` package tree, and instead points readers at the current sync/event migrations plus the thin `_exports_*.py` root facade.
- The archive-facing training docs now match the current compatibility story too: `archive/deprecated_training/README.md` and `ai-service/docs/MIGRATION_GUIDE.md` now point legacy readers at the archived `orchestrated_training.py` implementation plus the `app.training` root re-export, rather than telling them to import from the removed `app.training.orchestrated_training` module path.
- The remaining code-adjacent training references are now being tightened too: `app.training.unified_orchestrator`, `app.integration.p2p_integration`, `app.integration.model_lifecycle`, and the archived `orchestrated_training.py` docstring now all describe the archived `TrainingOrchestrator` path as a compatibility re-export from `app.training` instead of referring to the removed `app.training.orchestrated_training` module or presenting it as the default higher-level orchestration path.
- The broader AI-service docs are now being aligned too: `ai-service/docs/CONFIG_SOURCES.md`, `DEPRECATION_ROADMAP.md`, and `DEPRECATED_MODULES_MIGRATION.md` now refer to the archived training orchestrator path instead of treating `orchestrated_training.py` as an active module, and `docs/runbooks/MASTER_RUNBOOK_INDEX.md` now uses `get_sync_scheduler()` and `get_event_stats()` instead of the stale `EventRouter.get_instance()` pattern.
- The remaining planning/status docs are being pulled into the same migration story too: `DEPRECATION_TRACKER.md`, `CONSOLIDATION_STATUS_2025_12_19.md`, `STRATEGIC_IMPROVEMENT_PLAN_2025_12.md`, `architecture/ARCHITECTURE_NAMING.md`, and `CONSOLIDATION_ROADMAP.md` now refer to the archived training orchestrator path and `app.training` compatibility re-export instead of implying the removed `app/training/orchestrated_training.py` module still exists in the active package tree.
- The migration guide is now fully aligned with that same story too: `ai-service/docs/MIGRATION_GUIDE.md` no longer falls back to a stale `orchestrated_training.py` timeline row, and its training-orchestrator section now consistently frames the implementation as the archived `archive/deprecated_training/orchestrated_training.py` compatibility layer.
- The remaining sync and legacy-engine planning docs are now being corrected too: `DEPRECATION_TIMELINE.md` now uses the stable `app.game_engine` public API and the archived `archive/deprecated_ai/_game_engine_legacy.py` reality instead of steering readers toward `DefaultRulesEngine`, and `CONSOLIDATION_STATUS_2025_12_28.md` now treats `app.coordination.sync_coordinator.py` as the thin deprecated shim it is today rather than a future rename target.
- The legacy-engine archive docs are being brought into the same line too: `archive/deprecated_ai/README.md` now points active callers to `app.game_engine` instead of the nonexistent `app.rules.game_engine`, and `app/DEPRECATION_AUDIT.md` now describes `_game_engine_legacy.py` as an archived implementation behind the `app.game_engine` facade instead of a still-live module slated for direct deletion.
- The remaining rules/quality status docs are now following that same contract too: `RULES_ENGINE_SURFACE_AUDIT.md`, `CODEBASE_QUALITY_REPORT.md`, and `CONSOLIDATION_STATUS_2025_12_19.md` now treat `app.game_engine` as the stable public surface and `app/_game_engine_legacy.py` as a compatibility path to the archived implementation instead of calling the legacy file the primary engine.
- The remaining legacy-rules reference spec is aligned too: `ai-service/docs/specs/LEGACY_RULES_DIFF.md` now points readers at the archived game-engine implementation path via the compatibility symlink instead of presenting `app/_game_engine_legacy.py` as if it were the only canonical file location.
- The neural-net migration story is now being aligned the same way: `archive/deprecated_ai/README.md`, `app/DEPRECATION_AUDIT.md`, `app/ai/neural_net/__init__.py`, `DEPRECATION_TIMELINE.md`, `DEPRECATION_TRACKER.md`, and `CODEBASE_QUALITY_REPORT.md` now treat `app.ai.neural_net` as the supported public facade and `app.ai._neural_net_legacy` as the archived compatibility path, instead of implying the package itself is deprecated or that callers should target wildcard replacement paths.
- The remaining neural-net planning docs are now being pulled into the same contract too: `ai-service/docs/MIGRATION_GUIDE.md` now uses the real `create_model_for_board(...)` flow instead of a nonexistent `RingRiftNet`/`neural_net/network.py` path, and `STRANDED_FEATURES.md` plus `app/training/TRAIN_REFACTORING.md` now describe `_neural_net_legacy` as a compatibility drain behind `app.ai.neural_net` rather than a generic package migration target.
- The last live neural-net status/design docs are now being tightened too: `CONSOLIDATION_STATUS_2025_12_19.md` now treats `_neural_net_legacy` as an archived compatibility path behind `app.ai.neural_net`, and `TITANS_IMPLEMENTATION_PLAN.md` no longer targets a nonexistent `app/ai/ringrift_net.py` file, instead pointing architectural work at the supported `app.ai.neural_net` surface.
- The remaining explicit stale path in that TITANS plan is gone too: the note now points generally at older one-off model-file paths instead of repeating the removed `app/ai/ringrift_net.py` path, and the docs ratchet now forbids that filename entirely.
- The event-system docs are now being pulled onto the same package-surface contract too: live references in `COORDINATION_ARCHITECTURE.md`, `EVENT_WIRING_GUIDE.md`, `EVENT_WIRING_DIAGRAM.md`, `EVENT_SYSTEM_REFERENCE.md`, `EVENT_CATALOG.md`, `EVENT_PAYLOAD_SCHEMAS.md`, and `ADR-001-event-driven-architecture.md` now point at the `app.distributed.data_events` package layout (`event_types.py`, `event_bus.py`, `emit.py`) instead of the removed `app/distributed/data_events.py` file.
- The remaining event-integration guides are now aligned too: `INTEGRATION_CHECKLIST.md`, `coordination/EVENT_HANDLER_PATTERNS.md`, and the architecture copies of the event wiring/subscription docs now point at `app.distributed.data_events.event_types.py` and the package re-export path instead of the deleted `app/coordination/data_events.py`, while `p2p-manager-integration.md` now uses the supported `app.distributed` circuit-breaker API instead of the nonexistent `app/coordination/circuit_breaker.py` module.
- The broader event reference/runbook set is now being aligned as well: `EVENT_NAMING_CONVENTION.md`, `runbooks/EVENT_WIRING_VERIFICATION.md`, `runbooks/COORDINATION_EVENT_SYSTEM.md`, `PRIORITY_ACTION_PLAN_2025_12_26.md`, `audits/COORDINATOR_EVENT_AUDIT.md`, and `roadmaps/INTEGRATION_MIGRATION_PLAN.md` now point at `app/distributed/data_events/event_types.py` or the `app.distributed.data_events` package instead of the removed `app/distributed/data_events.py` file.

## Execution Protocol

Each autonomous batch should follow this loop:

1. identify one narrow cleanup seam
2. patch code/docs
3. run the smallest meaningful verification slice
4. commit with an intentional message
5. update this file if priorities or baselines changed

## Reassessment Triggers

Re-score the codebase after:

- each source-of-truth cleanup wave
- each major coordination cleanup wave
- each new quality gate added to CI/local automation

If future context is lost, resume from:

1. this file
2. [RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md)
3. [architecture/OVERVIEW.md](/Users/armand/Development/RingRift/docs/architecture/OVERVIEW.md)
4. [ai-service/scripts/README.md](/Users/armand/Development/RingRift/ai-service/scripts/README.md)
