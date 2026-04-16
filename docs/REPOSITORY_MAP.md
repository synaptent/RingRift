# Repository Map

This document is the shortest path through the RingRift repository for a new engineer. It is intentionally opinionated: it separates the supported path from the larger operational and historical surface.

## Top-Level Size Snapshot

Counts below are a tracked-file snapshot from April 10, 2026.

| Path          | Tracked Files | Approx. LOC | Read                                                                |
| ------------- | ------------: | ----------: | ------------------------------------------------------------------- |
| `src/`        |           404 |      161001 | TypeScript product and canonical rules surface                      |
| `ai-service/` |          3912 |     2888220 | Python AI service, training, coordination, scripts, tests, and docs |
| `scripts/`    |            79 |       23493 | Root operational wrappers and supported-path helpers                |
| `docs/`       |           326 |      140851 | Active documentation plus archived history                          |
| `tests/`      |           845 |      347810 | TypeScript/Jest/Playwright tests                                    |
| `archive/`    |            56 |       28441 | Historical root-level artifacts                                     |
| `monitoring/` |            25 |       14829 | Dashboards and monitoring assets                                    |
| `prisma/`     |            14 |         722 | Prisma schema and migrations                                        |

## Start Here

If you want to understand the project quickly, read these in order:

1. [README.md](/README.md)
2. [QUICKSTART.md](/QUICKSTART.md)
3. [RESULTS.md](/docs/RESULTS.md)
4. [ARCHITECTURE_OVERVIEW.md](/docs/ARCHITECTURE_OVERVIEW.md)
5. [DEVELOPER_GUIDE.md](/docs/DEVELOPER_GUIDE.md)
6. [INDEX.md](/docs/INDEX.md)

## Supported Path

### Canonical rules

- [`src/shared/engine`](/src/shared/engine)
- [`src/shared/types`](/src/shared/types)
- [`RULES_CANONICAL_SPEC.md`](/RULES_CANONICAL_SPEC.md)

This is the rules source of truth.

### Web product

- [`src/client`](/src/client)
- [`src/server`](/src/server)

This is the playable application surface.

### Python parity and AI service

- [`ai-service/app`](/ai-service/app)
- [`ai-service/scripts/check_ts_python_replay_parity.py`](/ai-service/scripts/check_ts_python_replay_parity.py)
- [`ai-service/scripts/ai_inference_smoke.py`](/ai-service/scripts/ai_inference_smoke.py)

This is the path that keeps training and inference aligned with the canonical TS engine.

### Supported training path

- [`scripts/run_proven_experiment.sh`](/scripts/run_proven_experiment.sh)
- [`ai-service/scripts/minimal_alphazero_loop.py`](/ai-service/scripts/minimal_alphazero_loop.py)
- [`ai-service/scripts/deploy_minimal_loops.sh`](/ai-service/scripts/deploy_minimal_loops.sh)
- [`ai-service/scripts/training_status.py`](/ai-service/scripts/training_status.py)

If your goal is to reproduce the current research results, this is the path to follow. For supported live canaries, use `deploy_minimal_loops.sh`; it preflights the minimal-loop test slice locally before restarting trainer nodes. During a run, treat `<work-dir>/progress.json` as the live stage-status file and `<work-dir>/metrics.jsonl` as the durable iteration log.

## Useful But Secondary

These surfaces are active and useful, but they are not the first stop for understanding the project:

- [`ai-service/scripts/README.md`](/ai-service/scripts/README.md)
- [`docs/operations`](/docs/operations)
- [`docs/runbooks`](/docs/runbooks)
- [`monitoring`](/monitoring)
- [`tests`](/tests)
- [`ai-service/tests`](/ai-service/tests)

## Legacy Or Operational Surface

These areas are real and increasingly better audited, but they are still not the shortest path to the core system:

- [`archive`](/archive)
- [`docs/archive`](/docs/archive)
- much of [`ai-service/app/coordination`](/ai-service/app/coordination)
- much of [`ai-service/scripts`](/ai-service/scripts) outside the supported path

The important distinction is not “delete versus keep.” It is “source of truth versus support infrastructure.”

## Recommended Reading Order By Goal

### I want to play the game

1. [README.md](/README.md)
2. [QUICKSTART.md](/QUICKSTART.md)
3. [`src/client`](/src/client)
4. [`src/server`](/src/server)

### I want to understand the rules implementation

1. [`RULES_CANONICAL_SPEC.md`](/RULES_CANONICAL_SPEC.md)
2. [`docs/rules/COMPLETE_RULES.md`](/docs/rules/COMPLETE_RULES.md)
3. [`src/shared/engine`](/src/shared/engine)
4. [`ai-service/scripts/check_ts_python_replay_parity.py`](/ai-service/scripts/check_ts_python_replay_parity.py)

### I want to reproduce the training evidence

1. [RESULTS.md](/docs/RESULTS.md)
2. [RESEARCH_SNAPSHOT.md](/docs/RESEARCH_SNAPSHOT.md)
3. [`scripts/run_proven_experiment.sh`](/scripts/run_proven_experiment.sh)
4. [`ai-service/scripts/minimal_alphazero_loop.py`](/ai-service/scripts/minimal_alphazero_loop.py)
5. [`docs/data/training_status.json`](/docs/data/training_status.json)

Use [CURRENT_STATUS.md](/docs/CURRENT_STATUS.md) only if you need the preserved April 10 owner-facing operational memo.

### I want to operate or extend the cluster

Read the supported path first, then move into:

- [`ai-service/scripts/deploy_minimal_loops.sh`](/ai-service/scripts/deploy_minimal_loops.sh)
- [`docs/operations`](/docs/operations)
- [`docs/runbooks`](/docs/runbooks)
- [`ai-service/scripts/training_status.py`](/ai-service/scripts/training_status.py)
- [`ai-service/scripts/training_dashboard.py`](/ai-service/scripts/training_dashboard.py)
- [`ai-service/scripts/fleet_health_check.py`](/ai-service/scripts/fleet_health_check.py)

For active trainer work directories, prefer `progress.json` for live stage status and `metrics.jsonl` for the durable per-iteration record before digging into broader daemon or cluster logs.

## Bottom Line

RingRift is easiest to understand if you treat it as four layers:

1. canonical TypeScript rules
2. playable web app
3. Python AI/parity mirror
4. minimal self-play training loop

The broader coordination and P2P surface is now meaningfully cleaner than it was at the start of Part 3, but it is still secondary to the supported path when you are orienting yourself.
