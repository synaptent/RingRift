# Part 3 Infrastructure Roadmap

Updated: April 10, 2026

This document records the Part 3 deep infrastructure improvement session so future work can resume without relying on chat history. The goal is to make RingRift easier to understand, verify, operate, and evolve without SSH archaeology or tribal knowledge.

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

## Current Baseline

- Branch: `main`.
- Known dirty file to leave untouched: `ai-service/archive/deprecated_ai/_game_engine_legacy.py`.
- P2P orchestrator baseline before Part 3: about 4,913 LOC with 14 extracted mixins.
- Coordination module baseline before Part 3: about 297K LOC across 314 files, with 20+ files over 3,000 LOC.
- Training status baseline: `hex8_2p` plateaued near 1968 Elo after 31 iterations and 6 promotions; `square8_2p` near 1602 Elo with 2 promotions but node health unstable; `square8_3p` near 1535 Elo and regressing under seat-fair evaluation; `square8_4p` still baseline near 1500 Elo.

## Roadmap

| Phase | Focus                                           | Target Outcome                                                                                                         | Status      |
| ----- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------- |
| 0     | Roadmap capture                                 | Durable document for the Part 3 goals and remaining work                                                               | Completed   |
| 1     | CI fix                                          | Supported-path workflow no longer fails when optional lint tools are absent; contract tests pass locally               | Completed   |
| 2     | P2P orchestrator                                | Extract state, peer discovery, job, process, HTTP, and game-count mixins; reduce `p2p_orchestrator.py` below 3,000 LOC | Completed   |
| 3     | Coordination module                             | Audit >3,000 LOC coordination files and extract repeated execution/lifecycle/strategy patterns                         | Completed   |
| 4     | Script consolidation                            | Inventory 602 scripts, archive deprecated scripts, and add a unified operational CLI                                   | Completed   |
| 5     | Training pipeline quality                       | Document minimal loop contracts, compare legacy behavior, add training pipeline contract tests                         | Completed   |
| 6     | Client code quality                             | Document extraction plans for large client files, run TypeScript checks, and reduce easy `as any` usage                | Completed   |
| 7     | Server code quality                             | Extract major route handlers and document server decomposition targets                                                 | Completed   |
| 8     | Test infrastructure                             | Remove empty tests, detect broken imports, add test-coverage meta-contracts, and clean conftest fixtures               | Completed   |
| 9     | Documentation cleanup                           | Archive stale 2025 docs and refresh current status, results, architecture, developer guide, and repository map         | Completed   |
| 10    | Type safety                                     | Audit `# type: ignore`, narrow bare `except`, add type-safety contracts                                                | Completed   |
| 11    | Config/environment cleanup                      | Audit legacy config inputs, refresh `.env.*.example` files, and remove stale deployment env surface                    | Completed   |
| 12    | Archive cleanup                                 | Audit active imports from archive modules and archive unused lambda scripts safely                                     | Completed   |
| 13    | Event system completion                         | Migrate remaining active `emit_event` calls to `safe_emit_event` and add canonical event contracts                     | Completed   |
| 14    | Large file decomposition: `app/ai` and `app/db` | Extract board encoding, MCTS tree logic, and replay validation modules with size contracts                             | In progress |
| 15    | Large file decomposition: `app/training`        | Extract training data pipeline, checkpointing, and Elo algorithms with size contracts                                  | Pending     |
| 16    | CI workflow consolidation                       | Add composite setup actions for Python AI and Node workflows                                                           | Pending     |
| 17    | Dead code and import cleanup                    | Detect unused app modules, circular imports, star imports, and obvious unused arguments                                | Pending     |
| 18    | Operational resilience                          | Add dead-loop restart and cluster health scripts plus supervisor heartbeat tests                                       | Pending     |
| 19    | Rules engine quality                            | Add parity coverage and rules completeness contracts across all supported board/player configs                         | Pending     |
| 20    | Final verification                              | Run Python tests, TypeScript checks, supported path checks, and update final architecture/audit docs                   | Pending     |

## Verification Rhythm

After each completed phase, run the targeted verification requested for the phase. The default phase gate is:

```bash
cd ai-service && PYTHONPATH=. python -m pytest tests/unit/ tests/contracts/ -x -q --timeout=120
```

For P2P-specific extraction, also run:

```bash
cd ai-service && PYTHONPATH=. python -m pytest tests/unit/p2p/ -x -q
```

For final verification, run:

```bash
cd ai-service && PYTHONPATH=. python -m pytest tests/ -x -q --timeout=300
npx tsc --noEmit
cd ai-service && PYTHONPATH=. python scripts/check_supported_path.py
```

## Resume Notes

If this session pauses before all phases are complete, resume from the first phase marked `Pending` or `In progress`. Do not infer completion from the roadmap alone; verify with git history, tests, and the referenced architecture documents.

## Progress Log

- Phase 0 completed: this roadmap was created so the Part 3 goals survive session context loss.
- Phase 1 completed: the optional Ruff contract now skips when Ruff is unavailable, and contract tests pass locally.
- Phase 2 completed: `scripts/p2p_orchestrator.py` was reduced below 3,000 LOC by extracting additional mixins; see `docs/P2P_DECOMPOSITION_PLAN.md`.
- Phase 3 completed: `COORDINATION_AUDIT.md` was created, the largest coordination modules were split, and the 2,500 LOC size contract now passes.
- Phase 3 extraction batch 1: `training_trigger_daemon.py` is 1,836 LOC and `daemon_manager.py` is 1,483 LOC after extracting execution and lifecycle mixins.
- Phase 3 extraction batch 2: `evaluation_daemon.py` is 1,563 LOC, `unified_queue_populator.py` is 1,470 LOC, and `data_pipeline_orchestrator.py` is 1,936 LOC after extracting execution/strategy/stage mixins.
- Phase 3 extraction batch 3: `curriculum_integration.py` is 504 LOC and `work_queue.py` is 1,873 LOC after extracting curriculum bridge/strategy helpers and work-queue storage helpers.
- Phase 3 extraction batch 4: `training_coordinator.py` is 1,862 LOC after extracting `TrainingJob` and slot/progress/status protocol helpers into `training_protocol.py`.
- Phase 3 extraction batch 5: the strict 2,500 LOC size contract required additional decomposition of `unified_health_manager.py`, `idle_resource_daemon.py`, `unified_distribution_daemon.py`, `tournament_daemon.py`, `event_router.py`, `event_emitters.py`, `coordination_bootstrap.py`, and `resource_optimizer.py`.
- Phase 3 size contract added: `tests/contracts/test_coordination_module_sizes.py` checks all `app/coordination/**/*.py` files and currently passes across 423 files.
- Phase 4 completed: the script inventory/archive sweep and unified operational CLI were added, with deprecated helpers moved out of the active surface.
- Phase 5 completed: `MINIMAL_LOOP_CONTRACT.md` and the training-infrastructure comparison work landed with supporting training pipeline contract coverage.
- Phase 6 completed: client decomposition plans were written, easy `as any` reductions were applied, and `npx tsc --noEmit` passed.
- Phase 7 completed: `SERVER_DECOMPOSITION_PLAN.md` was added, three large game route handlers were extracted, and the route layer was stabilized.
- Phase 8 completed: empty regression stubs and obsolete skip-only tests were removed, archive imports were migrated to active shims, new test-infrastructure contracts were added, timeout guards were added to hang-prone suites, and the phase gate passed with `32716 passed, 94 skipped`.
- Phase 9 completed: stale 2025 planning and historical documents were archived, `TODO.md` and the status/results/architecture/developer docs were refreshed to the April 10, 2026 state, `docs/data/training_status.json` was regenerated from `training_status.py --json --ssh`, and the phase gate passed with `32716 passed, 94 skipped`.
- Phase 9 gate stabilization: `tests/unit/monitoring/test_unified_health.py` now accepts zero-duration checks, matching the actual orchestrator behavior and preventing a flaky false negative during the phase gate.
- Phase 10 completed: broad bare-`except` handling was removed from the monitoring scripts, silent best-effort coordination catches were narrowed or logged, and `tests/contracts/test_type_safety.py` now enforces no bare `except` in `app/`, no `Any` in `app/rules` function signatures, and a bounded `# type: ignore` budget.
- Phase 10 typing baseline: `app/` type-ignore comments were reduced from `244` to `200`, with coded ignores increased from `68` to `124` and uncoded ignores reduced from `176` to `76`.
- Phase 10 verification: targeted contracts and focused unit tests passed, followed by the full gate at `32722 passed, 94 skipped`.
- Phase 11 completed: `cluster.yaml`, `cluster_nodes.yaml`, and `hyperparameters.json` were audited and confirmed as active compatibility inputs rather than dead files, `CONFIG_SOURCE_OF_TRUTH.md` was updated to reflect that status, and stale legacy references in the env/deployment surface were cleaned up.
- Phase 11 env cleanup: `.env.example`, `.env.production.example`, and `.env.staging.example` now reflect the current server/client variables, Sentry and heuristic-profile envs are documented, stale `SOCKET_PORT` / `MAX_CONCURRENT_CONNECTIONS` / `ORCHESTRATOR_ADAPTER_ENABLED` deployment exports were removed from compose files, and `vite-env.d.ts` now matches the actual `VITE_*` names used by the client.
- Phase 11 verification: `npx tsc --noEmit`, `python -m py_compile` on the touched AI-service modules, and `npx ts-node scripts/validate-deployment-config.ts` all passed; the required phase gate finished at `32722 passed, 94 skipped, 19 warnings`.
- Phase 12 completed: `ARCHIVE_IMPORT_AUDIT.md` documents the remaining live `archive.*` compatibility shims, `archive/lambda_scripts/` was relocated to `archive/deprecated_lambda/lambda_scripts/`, and `pytest.ini` now explicitly excludes `tests/archive` in addition to the archive-local `conftest.py` guard.
- Phase 12 gate stabilization: `TaskCoordinator` resource auto-recovery now prefers recent cached node-resource reports before falling back to local `psutil`, which stops false auto-resume on unrelated host metrics; the GPU selection tests now validate large-batch index bounds through CPU scalar min/max checks to avoid flaky MPS boolean reductions.
- Phase 12 verification: the required gate finished at `32722 passed, 94 skipped, 19 warnings` after the archive cleanup and the two gate-stability fixes above.
- Phase 13 completed: remaining active runtime `emit_event` paths in `scripts/p2p/**`, `scripts/p2p_orchestrator.py`, and `app/training/elo_recording.py` were migrated onto `app.coordination.event_emission_helpers.safe_emit_event`, and the targeted P2P/unit tests were updated to mock the consolidated helper path instead of the legacy router entrypoint.
- Phase 13 audit hardening: `ai-service/docs/architecture/EVENT_SYSTEM_AUDIT.md` now records that active coordination/P2P runtime emission is fully on the consolidated helper path, while compatibility/docstring-only `emit_event` references remain intentionally outside the supported-path contract.
- Phase 13 contracts: `ai-service/tests/contracts/test_event_system_canonical.py` now AST-scans `app/coordination` plus `scripts/p2p` for runtime `emit_event` calls and enforces subscriber coverage for the supported event catalog (`training_completed`, `evaluation_completed`, `model_promoted`, `sync_completed`, `new_games`, `curriculum_rebalanced`, `elo_velocity_changed`, `training_started`, `selfplay_complete`).
- Phase 13 verification: focused event-path tests passed at `165 passed`, and the required gate finished at `32728 passed, 94 skipped, 19 warnings`.
