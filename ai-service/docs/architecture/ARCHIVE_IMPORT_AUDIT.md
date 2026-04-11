# Archive Import Audit

Updated: April 10, 2026

This note records the active `archive.*` imports that still exist under
`ai-service/app/` after the Part 3 Phase 12 cleanup.

## Summary

- Result: no dead `archive.*` imports were found in active `app/` modules.
- Remaining imports are intentional compatibility shims or deprecated re-exports.
- `archive/lambda_scripts/` had no active code references and was relocated to
  `archive/deprecated_lambda/lambda_scripts/`.
- Archived tests are now excluded both by `tests/archive/conftest.py` and
  explicitly by `pytest.ini` via `norecursedirs = tests/archive`.

## Active Archive-Backed Imports

### `app/ai/ebmo_ai.py`

- Archive target: `archive.deprecated_ai.ebmo_ai`
- Status: intentional compatibility shim
- Why still active:
  - Imported by `app.ai.factory`
  - Used by historical tooling and tests that still construct `AIType.EBMO`
- Removal readiness: not ready; callers still exist

### `app/ai/ebmo_network.py`

- Archive target: `archive.deprecated_ai.ebmo_network`
- Status: intentional compatibility shim
- Why still active:
  - Imported by `app.routes.online_learning`
  - Used by `app.ai.ebmo_online_learner`, archive AI helpers, and data-generation scripts
- Removal readiness: not ready; runtime callers still exist

### `app/ai/gmo_ai.py`

- Archive target: `archive.deprecated_ai.gmo_ai`
- Status: intentional compatibility shim
- Why still active:
  - Imported by `app.ai.factory`
  - Used by `app.ai.gmo_mcts_hybrid`, training helpers, evaluation scripts, and ablation tooling
- Removal readiness: not ready; runtime and script callers still exist

### `app/ai/gmo_v2.py`

- Archive target: `archive.deprecated_ai.gmo_v2`
- Status: intentional compatibility shim
- Why still active:
  - Imported by `app.ai.factory`
  - Used by evaluation tooling that still exercises experimental GMOv2 paths
- Removal readiness: not ready; script callers still exist

### `app/ai/ig_gmo.py`

- Archive target: `archive.deprecated_ai.ig_gmo`
- Status: intentional compatibility shim
- Why still active:
  - Imported by `app.ai.factory`
  - Used by evaluation paths that still expose `AIType.IG_GMO`
- Removal readiness: not ready; script callers still exist

### `app/training/__init__.py`

- Archive target: `archive.deprecated_training.orchestrated_training`
- Status: deprecated re-export kept for backwards compatibility
- Why still active:
  - Exposes `TrainingOrchestrator*` symbols for older imports from `app.training`
  - The module suppresses deprecation warnings specifically to preserve those imports
- Removal readiness: not ready until the legacy import surface is removed

## Lambda Archive Audit

- Directory audited: `archive/lambda_scripts/`
- Active code references found: none
- Non-code references found:
  - `docs/EVENT_REFERENCE_AUTO.md`
- Action taken:
  - moved to `archive/deprecated_lambda/lambda_scripts/`
  - updated `EVENT_REFERENCE_AUTO.md` paths

## Archived Test Discovery

- Archive tests live under `tests/archive/`
- Existing guard:
  - `tests/archive/conftest.py` sets `collect_ignore_glob = ["**/*.py"]`
- Additional guard added:
  - `pytest.ini` now sets `norecursedirs = tests/archive`

This keeps archived tests out of normal discovery even if a future refactor
changes directory-local `conftest.py` behavior.
