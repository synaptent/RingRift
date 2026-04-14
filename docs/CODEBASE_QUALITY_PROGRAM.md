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
