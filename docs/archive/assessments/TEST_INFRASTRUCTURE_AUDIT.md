# Test Infrastructure Audit

Updated: April 10, 2026

## Scope

This audit covers the active Python test tree under `ai-service/tests/` for:

- dead or no-op test files
- direct `archive.*` imports from active tests
- broken `app.*` imports in active tests
- top-level module-to-test correspondence for `app/coordination/*.py` and `app/training/*.py`
- cross-`conftest.py` fixture shadowing risks

## Dead Test Files Removed

These files were still named `test_*.py` but contained only archived helper code and no
discoverable tests:

- `ai-service/tests/invariants/test_active_no_moves_movement_forced_elimination_regression.py`
- `ai-service/tests/invariants/test_active_no_moves_movement_fully_eliminated_regression.py`
- `ai-service/tests/invariants/test_active_no_moves_movement_placements_only_regression.py`

These regressions were already superseded by active invariant coverage referenced in each file.

This obsolete skip-only file was also removed because the implementation was consolidated and the
active coverage already lives in `test_handler_base.py`:

- `ai-service/tests/unit/coordination/test_event_subscription_mixin.py`

## Direct Archive Imports Removed

Active tests should not import from `archive.*` directly. They now use compatibility shims or the
current shared implementation instead:

- `ai-service/tests/test_ai_creation.py`
- `ai-service/tests/test_ebmo_ai.py`
- `ai-service/tests/test_gmo_ai.py`
- `ai-service/tests/test_gmo_v2_ai.py`
- `ai-service/tests/test_lane3_determinism.py`
- `ai-service/tests/unit/training/test_train_gmo.py`

## Broken Import Audit

After the cleanup above, the only remaining broken `app.*` imports are inside
`ai-service/tests/unit/distributed/pending/`, which is explicitly ignored by
`collect_ignore_glob = ["*.py"]` in that directory’s `conftest.py`.

Files in the ignored pending area still referencing unimplemented modules:

- `ai-service/tests/unit/distributed/pending/test_registries.py`

The stale NNUE registry test path was updated from `app.ai.nnue.registry` to the implemented
module path `app.ai.nnue_registry`.

## Conftest Shadowing

The only duplicate fixture names across `conftest.py` files were:

- `game_state_factory`
- `mock_daemon_manager`

Both duplicated definitions were unused in their narrower scopes and were removed, leaving the
root or unit-scoped fixtures as the canonical versions.

## Module Coverage Contract

`ai-service/tests/contracts/test_test_infrastructure.py` now enforces:

- active test files define at least one discoverable test
- active test files do not import `archive.*` directly
- active test files do not import missing `app.*` modules
- every top-level `app/coordination/*.py` module has a corresponding active test file
- every top-level `app/training/*.py` module has a corresponding active test file

For seven training helper modules without a same-stem test file, ownership is anchored via:

- `ai-service/tests/unit/training/test_training_module_smoke.py`

Those modules are:

- `auxiliary_tasks`
- `checkpointing`
- `distillation`
- `lr_finder`
- `opening_book`
- `pbt`
- `thread_integration`

## Timeout Guards Added

The audit also added module-level `pytest.mark.timeout(30)` guards to the
highest-risk active suites that exercise network, subprocess, socket, SSH, or
coordination loops:

- `ai-service/tests/integration/coordination/test_database_sync_manager.py`
- `ai-service/tests/integration/coordination/test_sync_router_integration.py`
- `ai-service/tests/integration/coordination/test_unified_node_health_daemon.py`
- `ai-service/tests/integration/p2p/test_swim_raft_endpoints.py`
- `ai-service/tests/unit/coordination/recovery/test_socket_leak_recovery.py`
- `ai-service/tests/unit/distributed/test_ssh_transport.py`
- `ai-service/tests/unit/scripts/test_master_loop_watchdog.py`
- `ai-service/tests/unit/scripts/test_p2p_orchestrator.py`
- `ai-service/tests/unit/scripts/test_training_probes.py`
