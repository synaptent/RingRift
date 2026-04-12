# Part 4 Quality Roadmap

Updated: April 12, 2026

This document records the next quality push after the Part 3 infrastructure session. The immediate goal is to move the codebase from "supported path is trustworthy" to "the active codebase is broadly verifiable and operationally sane" without spending time on archived or explicitly legacy paths with low return.

## Current Baseline

- Branch: `main`
- Known dirty file to leave untouched: `ai-service/archive/deprecated_ai/_game_engine_legacy.py`
- Supported-path verification is green:
  - `cd ai-service && PYTHONPATH=. python -m pytest tests/unit/ tests/contracts/ -x -q --timeout=120`
  - `npx tsc --noEmit`
  - `PYTHONPATH=ai-service bash scripts/check_supported_path.sh`
- Exhaustive AI-service verification is not yet green:
  - `cd ai-service && PYTHONPATH=. python -m pytest tests/ -x -q --timeout=300`
- Remaining active risk is concentrated in:
  - coordination/bootstrap side effects
  - non-hermetic tests
  - remaining large active modules
  - active compatibility boundaries around legacy code

## Constraints

- Do not modify `ai-service/scripts/minimal_alphazero_loop.py` or its support files.
- Do not modify `ai-service/config/distributed_hosts.yaml`.
- Do not modify database files under `data/` or `ai-service/data/`.
- Keep Python 3.10 compatibility: no `match`, no `datetime.UTC`, no `tomllib`.
- Run Python tests from `ai-service/` with `PYTHONPATH=.`.
- Follow `ai-service/AGENTS.md` for AI-service changes.
- Do not change `ai-service/app/main.py` `eval_mode` logic from commit `a1f8c80ff`.
- Leave `ai-service/archive/deprecated_ai/_game_engine_legacy.py` alone if it is the only dirty file.
- Commit every 3-5 tasks and push frequently.

## Roadmap

| Phase | Focus                                | Target Outcome                                                                                                  | Status      |
| ----- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------- | ----------- |
| 0     | Roadmap capture                      | Durable document for Part 4 goals, order, and verification                                                      | Completed   |
| 1     | Exhaustive suite stabilization       | `pytest tests/ -x -q --timeout=300` is clean or reduced to a tightly bounded residual list with fixes in flight | In progress |
| 2     | Coordination side-effect containment | Constructors/bootstrap paths stop doing real cluster/process work during unit-test setup                        | Pending     |
| 3     | Remaining active monolith reduction  | Split the largest active non-legacy files and add new size contracts                                            | Pending     |
| 4     | Test architecture cleanup            | Split giant tests, centralize builders/fixtures, and block live network/SSH from unit tests                     | Pending     |
| 5     | Legacy boundary hardening            | Make supported-path vs compatibility-path imports explicit and enforceable                                      | Pending     |
| 6     | Final trust pass                     | Full Python suite, TypeScript, and supported-path gates all green with refreshed docs                           | Pending     |

## Execution Order

1. Record the roadmap and keep it updated after each meaningful batch.
2. Stabilize the exhaustive AI-service suite before deeper refactors.
3. Contain coordination/bootstrap side effects once the concrete failing patterns are known.
4. Reduce the remaining large active modules only after the suite is stable enough to protect the changes.
5. Clean up oversized and integration-heavy tests so they stop hiding failures behind shared setup.
6. Harden the remaining legacy boundaries and rerun the full verification stack.

## Phase Detail

### Phase 1: Exhaustive Suite Stabilization

Primary command:

```bash
cd ai-service && PYTHONPATH=. python -m pytest tests/ -x -q --timeout=300
```

Focus areas:

- singleton leakage across tests
- persisted restart/supervisor state leaking across fixtures
- tests that do real SSH, filesystem, or cluster work during import/setup
- daemon-manager bootstrap paths that fan out unexpectedly
- timing-sensitive tests with brittle sleep windows

Definition of done:

- the full `tests/` gate passes, or
- a short, explicit residual-failure list exists in this document with root causes and next fixes already identified

### Phase 2: Coordination Side-Effect Containment

Candidate files:

- `ai-service/app/coordination/daemon_manager.py`
- `ai-service/app/coordination/daemon_manager_lifecycle.py`
- `ai-service/app/config/coordination_defaults.py`
- `ai-service/app/distributed/cluster_manifest.py`

Target:

- move heavyweight initialization behind explicit wiring/start boundaries
- make unit-test setup deterministic and local-only by default
- reduce ambient singleton/global state mutations during import

### Phase 3: Remaining Active Monolith Reduction

Highest-value targets at current baseline:

- `ai-service/app/config/coordination_defaults.py`
- `ai-service/app/distributed/cluster_manifest.py`
- `ai-service/app/distributed/data_events/emit.py`
- `ai-service/app/training/promotion_controller.py`

Target:

- reduce cognitive load in active supported-path modules
- create smaller helper modules with narrow ownership
- add/update line-count contracts after extractions

### Phase 4: Test Architecture Cleanup

Likely targets:

- `ai-service/tests/test_daemon_manager.py`
- `ai-service/tests/unit/coordination/test_event_emitters.py`
- `ai-service/tests/unit/coordination/test_cluster_transport.py`
- `ai-service/tests/unit/coordination/test_resource_optimizer.py`

Target:

- split fixture-heavy and scenario-heavy tests
- centralize daemon/cluster builders
- add a contract preventing live network/SSH in unit tests

### Phase 5: Legacy Boundary Hardening

Target:

- audit active imports of `archive.*` and `*_legacy.py`
- define supported-path compatibility boundaries explicitly
- remove dead runtime references and add import contracts where practical

### Phase 6: Final Trust Pass

Verification stack:

```bash
cd ai-service && PYTHONPATH=. python -m pytest tests/unit/ tests/contracts/ -x -q --timeout=120
cd ai-service && PYTHONPATH=. python -m pytest tests/ -x -q --timeout=300
npx tsc --noEmit
PYTHONPATH=ai-service bash scripts/check_supported_path.sh
```

Then update:

- `docs/architecture/PART4_QUALITY_ROADMAP.md`
- `docs/architecture/COORDINATION_AUDIT.md`
- any current-status or quality-status docs changed by the work

## Progress Log

- Phase 0 started: Part 4 roadmap created with the actual current baseline and remaining highest-ROI quality targets.
- Phase 0 completed: this roadmap now records the Part 4 execution order, target modules, and verification stack.
- Phase 1 progress: restored `scripts/run_tier_training_pipeline.py` compatibility for the supported tier-gating contract (`--run-dir`, preflight hooks, `training_report.json`, `status.json`) and refreshed `config/tier_training_pipeline.square8_2p.json` with explicit `num_games_override` values expected by the demo gate.
- Phase 1 progress: updated `tests/test_cluster_status_monitor.py` to mock the actual `SSHClient` boundary used by `ClusterMonitor` instead of the pre-migration `subprocess.run` path.
- Phase 1 progress: reduced `tests/test_benchmark_make_unmake.py::TestSearchModeEquivalence::test_both_modes_produce_valid_moves` to a bounded smoke test suitable for the default exhaustive suite while preserving the opt-in benchmark coverage.
- Phase 1 progress: replaced brittle source-string assertions in `tests/parity/test_victory_parity.py` with a behavioral stalemate-tiebreak check that explicitly exercises the deterministic fallback rung.
- Phase 1 progress: reduced `tests/test_heuristic_training_evaluation.py::test_evaluate_fitness_zero_profile_wiring_and_stats` to a bounded wiring check (`games_per_eval=2`, `max_moves=50`, seeded) so it still validates evaluator accounting without dominating the exhaustive gate.
- Phase 1 progress: aligned `app/coordination/pipeline_stages.py` orphan-game handlers with the current async contract so the daemon event-chain integration tests can await the real handler boundary.
- Phase 1 progress: refreshed `tests/test_quality_monitor_daemon.py` to the current `QualityResult`-based API and `event_router.publish` path instead of the older float-returning quality helpers.
- Phase 1 progress: made `scripts/lib/cluster.py::ClusterAutomation.check_p2p_status()` degrade to `"stopped"` on transport/protocol probe failures and rewrote the `ClusterManager` unit tests to mock `scripts.lib.hosts.get_active_hosts()` instead of asserting an obsolete hard-coded default fleet.
- Phase 1 progress: replaced stale hard-coded production Elo literals in `scripts/monitor_elo.py`, `scripts/weekly_gauntlet.py`, and `scripts/monitor_improvement.py` with imports from `app.config.thresholds`.
- Phase 1 progress: fixed `app/coordination/orchestrator_registry.py` heartbeat shutdown so registry tests no longer pay a 5-second join penalty on every `release_role()`. The heartbeat thread now uses an interruptible stop event instead of sleeping through shutdown.
- Phase 1 progress: restored `TrainingActivityDaemon._on_graceful_shutdown()` as a backward-compatible shim to the current `HandlerBase` `_on_stop()` lifecycle hook so shutdown-triggered final-sync integration coverage still exercises the real daemon behavior.
- Phase 1 targeted verification: `cd ai-service && PYTHONPATH=. python -m pytest tests/test_tier_pipeline_scripts.py tests/test_cluster_status_monitor.py tests/test_benchmark_make_unmake.py tests/parity/test_victory_parity.py -q --timeout=120` passed at `71 passed, 1 skipped`.
- Phase 1 targeted verification: `cd ai-service && PYTHONPATH=. python -m pytest tests/test_tier_pipeline_scripts.py tests/test_cluster_status_monitor.py tests/test_benchmark_make_unmake.py tests/parity/test_victory_parity.py tests/test_heuristic_training_evaluation.py -q --timeout=120` passed at `77 passed, 2 skipped`.
- Phase 1 targeted verification: `cd ai-service && PYTHONPATH=. python -m pytest tests/integration/coordination/test_daemon_event_chains.py -q -k orphan --timeout=120` passed at `5 passed, 15 deselected`.
- Phase 1 targeted verification: `cd ai-service && PYTHONPATH=. python -m pytest tests/test_quality_monitor_daemon.py tests/test_scripts_lib/test_lib_cluster.py tests/test_thresholds_usage.py -q --timeout=120` passed at `79 passed`.
- Phase 1 targeted verification: `cd ai-service && PYTHONPATH=. python -m pytest tests/unit/coordination/test_orchestrator_registry.py -q --timeout=120 --durations=20` passed at `85 passed in 1.25s`, down from the prior multi-minute slow zone dominated by 5-second registry shutdown waits.
- Phase 1 targeted verification: `cd ai-service && PYTHONPATH=. python -m pytest tests/integration/coordination/test_training_activity_coordination.py -q --timeout=120` passed at `13 passed`.
- Phase 1 exhaustive verification progress: `cd ai-service && PYTHONPATH=. python -m pytest tests/ -x -q --timeout=300` advanced from a hard stop at `3719 passed` on `tests/test_thresholds_usage.py::test_no_hardcoded_production_elo_threshold` to a later async/socket-heavy section after the threshold fixes, with no new deterministic failure yet isolated.
- Phase 1 exhaustive verification progress: the same gate now reaches `6981 passed, 37 skipped` before the next deterministic failure in `tests/integration/coordination/test_training_activity_coordination.py::TestTrainingActivityToSyncFlow::test_graceful_shutdown_triggers_final_sync`, and that shutdown-hook regression has now been fixed.
