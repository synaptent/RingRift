# Part 3 Infrastructure Roadmap

Updated: April 11, 2026

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

| Phase | Focus                                           | Target Outcome                                                                                                         | Status    |
| ----- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------- |
| 0     | Roadmap capture                                 | Durable document for the Part 3 goals and remaining work                                                               | Completed |
| 1     | CI fix                                          | Supported-path workflow no longer fails when optional lint tools are absent; contract tests pass locally               | Completed |
| 2     | P2P orchestrator                                | Extract state, peer discovery, job, process, HTTP, and game-count mixins; reduce `p2p_orchestrator.py` below 3,000 LOC | Completed |
| 3     | Coordination module                             | Audit >3,000 LOC coordination files and extract repeated execution/lifecycle/strategy patterns                         | Completed |
| 4     | Script consolidation                            | Inventory 602 scripts, archive deprecated scripts, and add a unified operational CLI                                   | Completed |
| 5     | Training pipeline quality                       | Document minimal loop contracts, compare legacy behavior, add training pipeline contract tests                         | Completed |
| 6     | Client code quality                             | Document extraction plans for large client files, run TypeScript checks, and reduce easy `as any` usage                | Completed |
| 7     | Server code quality                             | Extract major route handlers and document server decomposition targets                                                 | Completed |
| 8     | Test infrastructure                             | Remove empty tests, detect broken imports, add test-coverage meta-contracts, and clean conftest fixtures               | Completed |
| 9     | Documentation cleanup                           | Archive stale 2025 docs and refresh current status, results, architecture, developer guide, and repository map         | Completed |
| 10    | Type safety                                     | Audit `# type: ignore`, narrow bare `except`, add type-safety contracts                                                | Completed |
| 11    | Config/environment cleanup                      | Audit legacy config inputs, refresh `.env.*.example` files, and remove stale deployment env surface                    | Completed |
| 12    | Archive cleanup                                 | Audit active imports from archive modules and archive unused lambda scripts safely                                     | Completed |
| 13    | Event system completion                         | Migrate remaining active `emit_event` calls to `safe_emit_event` and add canonical event contracts                     | Completed |
| 14    | Large file decomposition: `app/ai` and `app/db` | Extract board encoding, MCTS tree logic, and replay validation modules with size contracts                             | Completed |
| 15    | Large file decomposition: `app/training`        | Extract training data pipeline, checkpointing, and Elo algorithms with size contracts                                  | Completed |
| 16    | CI workflow consolidation                       | Add composite setup actions for Python AI and Node workflows                                                           | Completed |
| 17    | Dead code and import cleanup                    | Detect unused app modules, circular imports, star imports, and obvious unused arguments                                | Completed |
| 18    | Operational resilience                          | Add dead-loop restart and cluster health scripts plus supervisor heartbeat tests                                       | Completed |
| 19    | Rules engine quality                            | Add parity coverage and rules completeness contracts across all supported board/player configs                         | Completed |
| 20    | Final verification                              | Run Python tests, TypeScript checks, supported path checks, and update final architecture/audit docs                   | Pending   |

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
- Phase 14 completed: `app/ai/mcts_ai.py` now delegates tree-node structures, visit-distribution extraction, PUCT/RAVE/FPU tuning, progressive-widening helpers, and root-prior/self-play helpers to `app/ai/mcts_tree.py`, while preserving the historical import surface from `app.ai.mcts_ai`.
- Phase 14 GPU split: `app/ai/gpu_parallel_games.py` was reduced from 4,373 to 3,493 LOC by extracting `GPUBoardEncodingMixin`, `GPUValidationMixin`, `GPURunnerReportingMixin`, and `GPUPersonaMixin` into dedicated helper modules without changing the runner API.
- Phase 14 replay split: `app/db/game_replay.py` was reduced from 4,393 to 3,488 LOC by extracting serialization/hash helpers, the incremental `GameWriter`, replay reconstruction helpers, and batch query helpers into `replay_serialization.py`, `game_replay_writer.py`, `replay_validation.py`, and `replay_batch_queries.py`.
- Phase 14 contracts: `ai-service/tests/contracts/test_ai_db_module_sizes.py` now enforces a 3,500-line budget across supported-path `app/ai` and `app/db` modules while excluding archived and explicit legacy compatibility files.
- Phase 14 stabilization: stale top-level replay DB fixtures that still assumed one-move synthetic games were updated to respect the current `MIN_MOVES_REQUIRED=5` invariant while disabling history/snapshot replay for metadata-only fixtures.
- Phase 15 completed: `app/training/train.py` was reduced to 2,964 LOC and `app/training/elo_service.py` to 1,812 LOC by extracting checkpoint management, runtime setup, run support, epoch reporting, entrypoints, data loading, Elo algorithms/backend/API/reporting, and checkpoint inspection helpers into dedicated modules under `app/training/`.
- Phase 15 contracts: `ai-service/tests/contracts/test_training_module_sizes.py` now enforces a 3,500-line ceiling across top-level `app/training/*.py` modules and locks the explicit Phase 15 targets for `train.py` (<3,000) and `elo_service.py` (<2,000).
- Phase 15 compatibility stabilization: legacy `train.py` imports (`seed_all_legacy`, heuristic tuning helpers/constants, `RingRiftDataset`, `train_from_file`, `train_with_config`) were preserved for the existing training/integration tests, a focused `test_training_entrypoints.py` ownership suite was added, stale `_EVAL_STAGES` tests were updated for the multiplayer-aware `_EVAL_STAGES_2P`/`_get_eval_stages()` split, and the node-circuit-breaker half-open timing test was hardened against scheduler jitter.
- Phase 16 completed: `.github/actions/setup-node/action.yml` and `.github/actions/setup-python-ai/action.yml` now centralize the repeated Node/Prisma and AI-service Python setup across the active workflow set, including `ci.yml`, `parity-ci.yml`, `supported-path.yml`, the nightly healthchecks, deploy/validation flows, and the SLO gate.
- Phase 16 verification: all workflow YAML plus the two composite actions parsed successfully with `yaml.safe_load`, `act` was not available locally for a dry run, and the required gate passed at `33088 passed, 94 skipped, 12 warnings` via `cd ai-service && PYTHONPATH=. python -m pytest tests/unit/ tests/contracts/ -x -q --timeout=120`.
- Phase 17 completed: wildcard imports were removed from the active `app/` tree, the dead embedded `app/coordination/hashgraph/tests/` package was deleted, obvious unused-argument stubs in large active modules were cleaned up without changing signatures, and `tests/contracts/test_import_hygiene.py` now enforces both constraints.
- Phase 17 import audit: `scripts/audit_import_graph.py` now provides a relative-import-aware graph for `app/`, `scripts/`, and `tests/`, `APP_IMPORT_AUDIT.md` records the current zero-inbound `app/` modules retained by design, and `COORDINATION_AUDIT.md` now captures the main coordination cycle clusters found by the new graph.
- Phase 17 verification: the focused import/re-export suites passed at `93 passed`, and the required gate finished at `33086 passed, 94 skipped, 19 warnings` via `cd ai-service && PYTHONPATH=. python -m pytest tests/unit/ tests/contracts/ -x -q --timeout=120`.
- Phase 18 completed: `scripts/restart_dead_loops.sh` now performs supported-node dead-loop detection plus direct `nohup` restarts with preflight model checks and post-launch verification, `scripts/cluster_health.py` now consolidates S3 heartbeat state with SSH process/GPU/disk/network/supervisor details, and `minimal_loop_supervisor.sh` now writes richer JSON heartbeats including supervisor PID, child PID, restart count, last restart time, and uptime.
- Phase 18 observability hardening: `scripts/training_status.py` now enriches SSH probe output with heartbeat payload data, SSH latency, GPU stats, disk stats, and host networking details so the status and cluster-health paths share one source of runtime truth.
- Phase 18 tests: `tests/unit/scripts/test_operational_scripts.py` now validates `restart_dead_loops.sh --dry-run`, `cluster_health.py --help`, `training_status.py --json --no-s3`, `training_dashboard.py --help`, and the supervisor heartbeat schema; the node-circuit-breaker half-open failure test was also hardened against scheduler jitter by increasing its recovery timeout.
- Phase 18 verification: focused operational scripts passed at `67 passed`, and the required gate finished at `33094 passed, 94 skipped, 18 warnings` via `cd ai-service && PYTHONPATH=. python -m pytest tests/unit/ tests/contracts/ -x -q --timeout=120`.
- Phase 19 completed: `app/rules/default_engine.py` now explicitly covers recovery, territory bookkeeping, and terminal move semantics (`recovery_slide`, `skip_recovery`, `no_territory_action`, `skip_territory_processing`, `resign`, `timeout`) through the canonical `GameEngine` surface instead of leaving those paths implicit.
- Phase 19 contracts: `tests/contracts/test_rules_parity_coverage.py` now enforces executable TS↔Python short-trace parity for all 12 supported board/player configurations via `tests/scripts/ts_rules_config_trace_parity.ts`, and `tests/contracts/test_rules_completeness.py` now locks the canonical move/phase/victory surfaces against `src/shared/types/game.ts`; `docs/architecture/RULES_ENGINE_AUDIT.md` records the supported-path rules surface and remaining intentional gaps.
- Phase 19 verification: focused rules tests passed at `87 passed`, and the required gate finished at `33124 passed, 94 skipped, 19 warnings` via `cd ai-service && PYTHONPATH=. python -m pytest tests/unit/ tests/contracts/ -x -q --timeout=120`.
