# Repository Map

This document is the shortest path through the RingRift repository for a new engineer.

The codebase is large and historically layered. The goal here is not to describe every file; it is to separate the supported path from the historical and operational surface.

## Start Here

If you want to understand the project quickly, read these in order:

1. [README.md](/Users/armand/Development/RingRift/README.md)
2. [docs/PROJECT_BRIEF.md](/Users/armand/Development/RingRift/docs/PROJECT_BRIEF.md)
3. [QUICKSTART.md](/Users/armand/Development/RingRift/QUICKSTART.md)
4. [docs/RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md)
5. [docs/ARCHITECTURE_OVERVIEW.md](/Users/armand/Development/RingRift/docs/ARCHITECTURE_OVERVIEW.md)

Then move to the code.

## Supported Code Path

### Canonical rules

- [`src/shared/engine`](/Users/armand/Development/RingRift/src/shared/engine)
- [`src/shared/types`](/Users/armand/Development/RingRift/src/shared/types)

This is the canonical game-rules implementation.

### Web product

- [`src/server`](/Users/armand/Development/RingRift/src/server)
- [`src/client`](/Users/armand/Development/RingRift/src/client)

This is the playable app.

### Python AI and parity

- [`ai-service/app`](/Users/armand/Development/RingRift/ai-service/app)
- [`ai-service/scripts/check_ts_python_replay_parity.py`](/Users/armand/Development/RingRift/ai-service/scripts/check_ts_python_replay_parity.py)

This is the AI service and the parity surface that keeps training data trustworthy.

### Supported training path

- [`scripts/run_proven_experiment.sh`](/Users/armand/Development/RingRift/scripts/run_proven_experiment.sh)
- [`ai-service/scripts/minimal_alphazero_loop.py`](/Users/armand/Development/RingRift/ai-service/scripts/minimal_alphazero_loop.py)

If your goal is to reproduce the reported research results, this is the path to follow.

## Useful But Secondary

These areas are current and useful, but they are not the fastest way to understand the repo:

- [`ai-service/README.md`](/Users/armand/Development/RingRift/ai-service/README.md)
- [`ai-service/scripts/README.md`](/Users/armand/Development/RingRift/ai-service/scripts/README.md)
- [`docs/production`](/Users/armand/Development/RingRift/docs/production)
- [`docs/runbooks`](/Users/armand/Development/RingRift/docs/runbooks)
- [`monitoring`](/Users/armand/Development/RingRift/monitoring)
- [`tests`](/Users/armand/Development/RingRift/tests)
- [`ai-service/tests`](/Users/armand/Development/RingRift/ai-service/tests)

## Historical Or Operationally Specialized Surface

These areas are real, but should not be your first stop:

- [`archive`](/Users/armand/Development/RingRift/archive)
- [`docs/archive`](/Users/armand/Development/RingRift/docs/archive)
- many cluster automation and operational scripts in [`ai-service/scripts`](/Users/armand/Development/RingRift/ai-service/scripts)
- assistant-facing memory files such as `CLAUDE.md` and `CLAUDE.local.md`

The repository contains a large number of scripts for cluster operations, monitoring, and historical experimentation. Those scripts are valuable operationally, but they are not all equally current or necessary for reproducing the core findings.

## Recommended Reading Order By Goal

### I want to play the game

1. [README.md](/Users/armand/Development/RingRift/README.md)
2. [QUICKSTART.md](/Users/armand/Development/RingRift/QUICKSTART.md)
3. [`src/client`](/Users/armand/Development/RingRift/src/client)
4. [`src/server`](/Users/armand/Development/RingRift/src/server)

### I want to understand the rules implementation

1. [`RULES_CANONICAL_SPEC.md`](/Users/armand/Development/RingRift/RULES_CANONICAL_SPEC.md)
2. [`docs/rules/COMPLETE_RULES.md`](/Users/armand/Development/RingRift/docs/rules/COMPLETE_RULES.md)
3. [`src/shared/engine`](/Users/armand/Development/RingRift/src/shared/engine)
4. [`ai-service/scripts/check_ts_python_replay_parity.py`](/Users/armand/Development/RingRift/ai-service/scripts/check_ts_python_replay_parity.py)

### I want to reproduce the training results

1. [docs/RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md)
2. [`scripts/run_proven_experiment.sh`](/Users/armand/Development/RingRift/scripts/run_proven_experiment.sh)
3. [`ai-service/scripts/minimal_alphazero_loop.py`](/Users/armand/Development/RingRift/ai-service/scripts/minimal_alphazero_loop.py)
4. [`ai-service/data/proven_experiments`](/Users/armand/Development/RingRift/ai-service/data/proven_experiments)

### I want to operate or extend the cluster

Read the supported path first, then move into:

- [`docs/operations`](/Users/armand/Development/RingRift/docs/operations)
- [`docs/runbooks`](/Users/armand/Development/RingRift/docs/runbooks)
- [`ai-service/scripts`](/Users/armand/Development/RingRift/ai-service/scripts)

## Bottom Line

RingRift is easiest to understand if you treat it as four layers:

1. canonical TypeScript rules
2. playable web app
3. Python AI/parity mirror
4. minimal self-play training loop

Everything else should be considered support, operations, or historical context until you need it.
