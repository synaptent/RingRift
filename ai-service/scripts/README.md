# AI Service Scripts

This directory contains a large amount of training, parity, export, cluster, and operational tooling.

It is useful, but it is not a single coherent product surface. If you are new to RingRift, do not start by reading every script here.

Start with the supported path, then move into the operational surface only if you need it.

## Start Here

If your goal is to understand or reproduce the current research result, read these in order:

1. [README.md](/Users/armand/Development/RingRift/README.md)
2. [QUICKSTART.md](/Users/armand/Development/RingRift/QUICKSTART.md)
3. [docs/PROJECT_BRIEF.md](/Users/armand/Development/RingRift/docs/PROJECT_BRIEF.md)
4. [docs/RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md)
5. [scripts/run_proven_experiment.sh](/Users/armand/Development/RingRift/scripts/run_proven_experiment.sh)

Then come back here for script-level detail.

## Supported For External Readers

These are the scripts that matter most for understanding the current project.

### Reproduce the reported experiments

- [`/Users/armand/Development/RingRift/scripts/run_proven_experiment.sh`](/Users/armand/Development/RingRift/scripts/run_proven_experiment.sh)
  - Top-level wrapper for the supported experiment presets.
  - Best first stop for anyone trying to reproduce the published `hex8_2p` or `square8_2p` results.

- [`minimal_alphazero_loop.py`](/Users/armand/Development/RingRift/ai-service/scripts/minimal_alphazero_loop.py)
  - Supported self-play / train / evaluate loop for the current research path.
  - This is the narrow experiment harness behind the reproducible presets.

### Verify TypeScript ↔ Python parity

- [`check_ts_python_replay_parity.py`](/Users/armand/Development/RingRift/ai-service/scripts/check_ts_python_replay_parity.py)
  - Main parity harness.
  - Use this to verify that the Python mirror still matches the canonical TypeScript engine on replayed games.

- [`diff_state_bundle.py`](/Users/armand/Development/RingRift/ai-service/scripts/diff_state_bundle.py)
  - Focused parity debugging tool for a single emitted state bundle.

### Work with canonical replay data

- [`generate_canonical_selfplay.py`](/Users/armand/Development/RingRift/ai-service/scripts/generate_canonical_selfplay.py)
  - Preferred canonical self-play generator and gate.

- [`check_canonical_phase_history.py`](/Users/armand/Development/RingRift/ai-service/scripts/check_canonical_phase_history.py)
  - Validates canonical phase-history semantics for a replay database.

- [`export_replay_dataset.py`](/Users/armand/Development/RingRift/ai-service/scripts/export_replay_dataset.py)
  - Preferred replay-to-dataset export path for current training work.

## Supported Workflow By Goal

### I want to prove the training pipeline works

```bash
./scripts/run_proven_experiment.sh square8_2p --print-only
./scripts/run_proven_experiment.sh square8_2p --iterations 10
```

### I want to inspect or debug parity

```bash
cd ai-service
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py --db <path-to-db>
```

### I want to validate replay history before training on a DB

```bash
cd ai-service
PYTHONPATH=. python scripts/check_canonical_phase_history.py --db <path-to-db>
```

## Operations And Cluster Surface

These scripts are real and actively useful for cluster operations, but they are not the best starting point for understanding the project.

### Coordinator / orchestration

- [`master_loop.py`](/Users/armand/Development/RingRift/ai-service/scripts/master_loop.py)
  - Coordinator-oriented multi-board orchestration loop.
  - Useful for long-running fleet operations.
  - Not the recommended first entrypoint for reproducing the published results.

- [`p2p_orchestrator.py`](/Users/armand/Development/RingRift/ai-service/scripts/p2p_orchestrator.py)
  - Cluster node coordination and job orchestration.

- [`node_resilience.py`](/Users/armand/Development/RingRift/ai-service/scripts/node_resilience.py)
  - Supervisor / fallback behavior for nodes.

### Sync and export

- [`unified_data_sync.py`](/Users/armand/Development/RingRift/ai-service/scripts/unified_data_sync.py)
  - Current sync entrypoint.

- [`distributed_export.py`](/Users/armand/Development/RingRift/ai-service/scripts/distributed_export.py)
  - Parallel export tooling for larger jobs.

- [`update_cluster_code.py`](/Users/armand/Development/RingRift/ai-service/scripts/update_cluster_code.py)
  - Cluster rollout helper.

If you are operating a live fleet, read [docs/operations](/Users/armand/Development/RingRift/docs/operations) and [docs/runbooks](/Users/armand/Development/RingRift/docs/runbooks) before using these directly.

## Secondary Or Historical Surface

This directory also contains many scripts that reflect older experiments, alternate pipelines, or specialized operational needs.

Examples:

- tier-based or alternate training pipelines such as `run_tier_training_pipeline.py`
- older automation wrappers such as `auto_training_pipeline.py`
- board-specific or one-off training flows such as `hex8_training_pipeline.py`
- Vast.ai or cluster-specific utilities
- archived self-play helpers under `archive/`

Those files can still be useful, but they should not be mistaken for the main supported research path.

## Recommended Reading Order

If you need more than the supported path, use this order:

1. [`minimal_alphazero_loop.py`](/Users/armand/Development/RingRift/ai-service/scripts/minimal_alphazero_loop.py)
2. [`check_ts_python_replay_parity.py`](/Users/armand/Development/RingRift/ai-service/scripts/check_ts_python_replay_parity.py)
3. [`generate_canonical_selfplay.py`](/Users/armand/Development/RingRift/ai-service/scripts/generate_canonical_selfplay.py)
4. [`export_replay_dataset.py`](/Users/armand/Development/RingRift/ai-service/scripts/export_replay_dataset.py)
5. `master_loop.py` and cluster scripts only if you are dealing with fleet operations

## Bottom Line

Treat this directory as three layers:

1. supported experiment and parity tools
2. operational cluster tooling
3. historical or specialized scripts

If you follow that framing, the directory is much easier to navigate and much less misleading.
