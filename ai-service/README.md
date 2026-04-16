# RingRift AI Service

The AI service is the Python side of RingRift. It serves inference endpoints, mirrors the canonical TypeScript rules for parity-sensitive workflows, and contains the training and replay-validation tooling used for the project’s published results.

For a project-level overview, start with [README.md](/README.md). For the current evidence, see [docs/RESULTS.md](/docs/RESULTS.md).

If you want the shortest project summary first, read [docs/PROJECT_BRIEF.md](/docs/PROJECT_BRIEF.md).

## Supported Paths

### 1. Inference service

```bash
cd ai-service
./setup.sh
./run.sh
```

Useful endpoints:

- `GET /health`
- `GET /docs`
- `POST /ai/move`
- `POST /ai/evaluate`
- `POST /ai/choice`

### 2. Reproduce the proven training path

From the repo root:

```bash
./scripts/run_proven_experiment.sh hex8_2p
./scripts/run_proven_experiment.sh square8_2p
```

Those wrappers call [`scripts/minimal_alphazero_loop.py`](/ai-service/scripts/minimal_alphazero_loop.py), which is the supported training engine for the published results.

For supported trainer canaries on the GH200 fleet, deploy with:

```bash
cd ai-service
bash scripts/deploy_minimal_loops.sh --dry-run
bash scripts/deploy_minimal_loops.sh
```

That deploy path now runs a local minimal-loop preflight before it restarts trainer nodes. Live stage state is written to each work directory's `progress.json`, while durable iteration history is written to `metrics.jsonl`.

### 3. Verify TypeScript ↔ Python parity

```bash
cd ai-service
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py --db <path-to-db>
```

This is the main trust boundary for training data quality.

## Key Files

- [`app/main.py`](/ai-service/app/main.py): FastAPI app
- [`app/README.md`](/ai-service/app/README.md): package map for the Python application tree
- [`app/game_engine`](/ai-service/app/game_engine): Python rules mirror
- [`app/training`](/ai-service/app/training): training stack
- [`scripts/minimal_alphazero_loop.py`](/ai-service/scripts/minimal_alphazero_loop.py): supported minimal training loop
- [`scripts/check_ts_python_replay_parity.py`](/ai-service/scripts/check_ts_python_replay_parity.py): replay parity harness
- [`scripts/README.md`](/ai-service/scripts/README.md): curated scripts index for supported vs ops-only paths
- [`TRAINING_DATA_REGISTRY.md`](/ai-service/TRAINING_DATA_REGISTRY.md): data provenance and status

## Supported vs Secondary

### Supported

- inference service
- parity tooling
- minimal training loop
- canonical checkpoints and replay data workflows

### Secondary

- broader cluster coordination scripts
- production deployment helpers
- AI calibration and ladder tooling

### Historical or Specialized

- deprecated or archived scripts
- older orchestration paths not needed for reproducing the main results

The repository still contains a large amount of operations and cluster code. That code is useful, but it is not the shortest path to understanding or reproducing the project’s main research outcomes.
