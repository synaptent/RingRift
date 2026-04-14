# RingRift Architecture Overview

This document is the external-facing architecture guide for the current RingRift codebase. It focuses on the supported path used to produce the reported results, not every historical or operational subsystem in the repository.

## System At A Glance

```text
React client ──> Node/Express server ──> shared TypeScript rules engine
                                          │
                                          ├─> browser sandbox / backend validation
                                          │
                                          └─> Python AI service parity mirror
                                                   │
                                                   ├─> inference endpoints
                                                   ├─> replay validation
                                                   └─> minimal AlphaZero loop
```

## The Core Design

### 1. TypeScript is the rules source of truth

The canonical game rules live under [`src/shared/engine`](/Users/armand/Development/RingRift/src/shared/engine) and the related shared types under [`src/shared/types`](/Users/armand/Development/RingRift/src/shared/types).

That engine is used in two places:

- the web/backend product path
- the parity target for the Python AI service

When the TypeScript engine and Python diverge, the TypeScript engine wins and Python must be fixed to match.

### 2. Python mirrors the rules for AI and training

The Python AI service lives under [`ai-service/app`](/Users/armand/Development/RingRift/ai-service/app). Its responsibilities are:

- serve AI moves and evaluations
- reconstruct and validate replay data
- mirror the canonical rules closely enough for training to be valid

The key guardrail is TS↔Python replay parity, checked with tools such as [`ai-service/scripts/check_ts_python_replay_parity.py`](/Users/armand/Development/RingRift/ai-service/scripts/check_ts_python_replay_parity.py).

### 3. The supported training engine is the minimal loop

The supported training path is [`ai-service/scripts/minimal_alphazero_loop.py`](/Users/armand/Development/RingRift/ai-service/scripts/minimal_alphazero_loop.py).

That script owns the end-to-end self-improvement cycle on a single worker:

1. selfplay
2. export to NPZ
3. train candidate
4. evaluate candidate vs best
5. promote or reject

The minimal loop is the training engine used for the current reported results. It is also the path wrapped by [`scripts/run_proven_experiment.sh`](/Users/armand/Development/RingRift/scripts/run_proven_experiment.sh).

For supported live canaries, the operator entrypoint is [`ai-service/scripts/deploy_minimal_loops.sh`](/Users/armand/Development/RingRift/ai-service/scripts/deploy_minimal_loops.sh). That rollout path preflights `tests/unit/scripts/test_minimal_alphazero_loop.py` locally before it restarts trainer nodes unless an operator explicitly bypasses the guard. During a run, treat `<work-dir>/progress.json` as the live stage-status file and `<work-dir>/metrics.jsonl` as the durable iteration log.

### 4. The broader coordinator is support infrastructure

The repository also contains a large amount of orchestration, cluster, daemon, and P2P machinery, especially under [`ai-service/scripts`](/Users/armand/Development/RingRift/ai-service/scripts) and [`ai-service/app/coordination`](/Users/armand/Development/RingRift/ai-service/app/coordination).

That infrastructure exists for:

- cluster operations
- model distribution
- monitoring
- historical experiments

It is not required to understand the core system or reproduce the main training results.

It should also not be treated as disposable. The current infrastructure strategy is to keep the minimal loop as the reproducible proof harness while reusing audited pieces of the broader coordinator/P2P stack where they satisfy the current rules, data, and evaluation contracts. See [docs/architecture/TRAINING_INFRASTRUCTURE_STRATEGY.md](/Users/armand/Development/RingRift/docs/architecture/TRAINING_INFRASTRUCTURE_STRATEGY.md).

As of April 13, 2026:

- the P2P orchestrator main file is down to `2591` LOC
- its behavior is split across `21` mixins totaling `12618` LOC
- the coordination directory is now protected by file-size and contract tests
- the repo has first-class status/health tooling via `training_status.py`, `training_dashboard.py`, and the product smoke gate

## Supported Path For New Readers

If you are approaching RingRift as an engineer or researcher, the most useful path through the repo is:

1. [README.md](/Users/armand/Development/RingRift/README.md)
2. [QUICKSTART.md](/Users/armand/Development/RingRift/QUICKSTART.md)
3. [docs/RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md)
4. [`src/shared/engine`](/Users/armand/Development/RingRift/src/shared/engine)
5. [`ai-service/scripts/minimal_alphazero_loop.py`](/Users/armand/Development/RingRift/ai-service/scripts/minimal_alphazero_loop.py)
6. [`ai-service/scripts/deploy_minimal_loops.sh`](/Users/armand/Development/RingRift/ai-service/scripts/deploy_minimal_loops.sh)
7. [`docs/architecture/MINIMAL_LOOP_CONTRACT.md`](/Users/armand/Development/RingRift/docs/architecture/MINIMAL_LOOP_CONTRACT.md)
8. [`ai-service/scripts/check_ts_python_replay_parity.py`](/Users/armand/Development/RingRift/ai-service/scripts/check_ts_python_replay_parity.py)
9. [`ai-service/scripts/training_status.py`](/Users/armand/Development/RingRift/ai-service/scripts/training_status.py)
10. [`docs/RESEARCH_SNAPSHOT.md`](/Users/armand/Development/RingRift/docs/RESEARCH_SNAPSHOT.md)

## Current Training Shape

The published results come from a small set of board/player configurations rather than every possible setup in the repo.

As of April 13, 2026:

- `hex8_2p` is the strongest result at `1979.8` and remains the main headline path
- `square8_2p` is the second clear improvement path at `1601.8`
- `square8_3p` now has one corrected-threshold multiplayer promotion, but the evidence is still weak
- `square8_4p` is still baseline and unproven

Large-board and some multiplayer configurations remain slower and less mature.

## Trust Boundaries

There are three main trust boundaries in the codebase:

### Rules boundary

- TypeScript rules engine is authoritative
- Python must match it

### Training-data boundary

- replay data is only trustworthy if the Python mirror is parity-correct
- results are only trustworthy if the training harness is configured correctly

### Product boundary

- the web app can function independently of the full training cluster
- production inference and research training are related, but they are not the same operational path

## What To Ignore On First Read

These areas are real, but they are not the best entry point:

- `archive/`
- `docs/archive/`
- most cluster-wide orchestration scripts under `ai-service/scripts`
- internal planning and assistant-facing notes such as `CLAUDE.md`

Use them later if you need operational or historical context. They are not the shortest path to understanding the project.

## Current Architectural Read

The architecture is now split into two explicit lanes:

- the **supported lane**, centered on the shared TS engine, Python parity mirror, and minimal training loop
- the **legacy-but-rehabilitated lane**, centered on coordination, P2P, and operational tooling that is being audited and decomposed rather than deleted

That distinction matters. The broader infrastructure is becoming usable again, but the project should still present the supported lane as the source of truth for scientific claims and reproducibility.
