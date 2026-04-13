# Developer Guide

This is the quickest practical path for a developer who wants to run RingRift locally, understand what matters, and avoid getting lost in the legacy operational surface.

## What To Read First

1. [README.md](/Users/armand/Development/RingRift/README.md)
2. [QUICKSTART.md](/Users/armand/Development/RingRift/QUICKSTART.md)
3. [ARCHITECTURE_OVERVIEW.md](/Users/armand/Development/RingRift/docs/ARCHITECTURE_OVERVIEW.md)
4. [RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md)
5. [REPOSITORY_MAP.md](/Users/armand/Development/RingRift/docs/REPOSITORY_MAP.md)
6. [INDEX.md](/Users/armand/Development/RingRift/docs/INDEX.md)

## Start The Product

Install dependencies:

```bash
npm install
cd ai-service && ./setup.sh
```

Run the web app:

```bash
npm run dev
```

Run the Python AI service separately if needed:

```bash
cd ai-service && ./run.sh
```

Useful split commands:

```bash
npm run dev:server
npm run dev:client
cd ai-service && PYTHONPATH=. uvicorn app.main:app --reload --port 8001
```

## Core Rules And Parity

Treat these as the main trust chain:

- [`RULES_CANONICAL_SPEC.md`](/Users/armand/Development/RingRift/RULES_CANONICAL_SPEC.md)
- [`src/shared/engine`](/Users/armand/Development/RingRift/src/shared/engine)
- [`src/shared/types`](/Users/armand/Development/RingRift/src/shared/types)
- [`ai-service/scripts/check_ts_python_replay_parity.py`](/Users/armand/Development/RingRift/ai-service/scripts/check_ts_python_replay_parity.py)

TypeScript is the executable rules source of truth. Python must match it.

## Supported Training Path

The supported research path is the minimal loop:

- [`scripts/run_proven_experiment.sh`](/Users/armand/Development/RingRift/scripts/run_proven_experiment.sh)
- [`ai-service/scripts/minimal_alphazero_loop.py`](/Users/armand/Development/RingRift/ai-service/scripts/minimal_alphazero_loop.py)

Useful status and validation commands:

```bash
npm run training:status -- --ssh
npm run training:dashboard
npm run training:validate-db -- <path-to-db>
npm run training:provenance -- <path-to-db>
```

Do not treat the broader legacy coordinator as the research source of truth unless a doc explicitly says that path is supported.

## Product Smoke And Diagnostics

Run the end-to-end product smoke:

```bash
npm run smoke:product
```

Run the supported-path validation:

```bash
cd ai-service && PYTHONPATH=. python scripts/check_supported_path.py
```

Run the Python AI inference smoke directly:

```bash
cd ai-service && PYTHONPATH=. python scripts/ai_inference_smoke.py
```

## Test Commands

Fast local Python gate:

```bash
cd ai-service && PYTHONPATH=. python -m pytest tests/unit/ tests/contracts/ -x -q --timeout=120
```

Full Python suite:

```bash
cd ai-service && PYTHONPATH=. python -m pytest tests/ -x -q --timeout=300
```

TypeScript check:

```bash
npx tsc --noEmit
```

Root test entry points:

```bash
npm test
npm run test:core
npm run test:e2e
```

## Where To Work

If you are changing rules:

- update TS first under [`src/shared/engine`](/Users/armand/Development/RingRift/src/shared/engine)
- then update Python mirrors under [`ai-service/app`](/Users/armand/Development/RingRift/ai-service/app)
- then rerun parity/contract checks

If you are changing training or replay logic:

- prefer the minimal loop and the parity/validation scripts
- keep data provenance explicit
- avoid pulling legacy DBs or models back into the supported path

If you are changing operational tooling:

- start with `training_status.py`, `training_dashboard.py`, `fleet_health_check.py`, and the runbooks
- keep the supported minimal path stable

## Docs That Matter

- [RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md)
- [RESEARCH_SNAPSHOT.md](/Users/armand/Development/RingRift/docs/RESEARCH_SNAPSHOT.md)
- [ARCHITECTURE_OVERVIEW.md](/Users/armand/Development/RingRift/docs/ARCHITECTURE_OVERVIEW.md)
- [REPOSITORY_MAP.md](/Users/armand/Development/RingRift/docs/REPOSITORY_MAP.md)
- [PART3_INFRASTRUCTURE_ROADMAP.md](/Users/armand/Development/RingRift/docs/architecture/PART3_INFRASTRUCTURE_ROADMAP.md)
- [TRAINING_INFRASTRUCTURE_STRATEGY.md](/Users/armand/Development/RingRift/docs/architecture/TRAINING_INFRASTRUCTURE_STRATEGY.md)

Use [CURRENT_STATUS.md](/Users/armand/Development/RingRift/docs/CURRENT_STATUS.md) only if you specifically need the preserved April 10 owner memo rather than the current supported state.

## Practical Rule

If you are unsure whether a subsystem is the supported path, assume it is not until one of the docs above says otherwise.
