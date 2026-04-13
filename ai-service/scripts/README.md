# AI Service Scripts

This directory contains hundreds of scripts. Only a small subset is part of the supported operational surface.

If you are new to the codebase, start with the scripts below and treat most `analyze_*`, `benchmark_*`, and `debug_*` files as investigative tools rather than supported entrypoints.

## Essential Scripts

### Fleet runtime

- [`minimal_alphazero_loop.py`](/Users/armand/Development/RingRift/ai-service/scripts/minimal_alphazero_loop.py)
  - Canonical trainer loop for self-play, train, and evaluate on a single config.

- [`policy_selfplay_worker.py`](/Users/armand/Development/RingRift/ai-service/scripts/policy_selfplay_worker.py)
  - Dedicated policy-bearing Gumbel selfplay worker for role-based fleet nodes.

- [`ingest_policy_selfplay.py`](/Users/armand/Development/RingRift/ai-service/scripts/ingest_policy_selfplay.py)
  - Validates, converts, and stages policy selfplay JSONL into trainer supplemental NPZ shards.

- [`deploy_training_service.sh`](/Users/armand/Development/RingRift/ai-service/scripts/deploy_training_service.sh)
  - Role-aware systemd deployment for trainer, selfplay-worker, evaluator, and P2P services.

- [`autonomy_fleet_check.py`](/Users/armand/Development/RingRift/ai-service/scripts/autonomy_fleet_check.py)
  - Fleet health probe for the current role-based GH200 deployment.

- [`fleet_health_check.py`](/Users/armand/Development/RingRift/ai-service/scripts/fleet_health_check.py)
  - Broader fleet health diagnostics across hosts and services.

- [`p2p_orchestrator.py`](/Users/armand/Development/RingRift/ai-service/scripts/p2p_orchestrator.py)
  - P2P control plane for model sync, job coordination, and node health.

### Canonical data and parity

- [`generate_canonical_selfplay.py`](/Users/armand/Development/RingRift/ai-service/scripts/generate_canonical_selfplay.py)
  - Canonical selfplay generation with parity and history gating.

- [`run_canonical_selfplay_parity_gate.py`](/Users/armand/Development/RingRift/ai-service/scripts/run_canonical_selfplay_parity_gate.py)
  - Batch parity gate for canonical selfplay outputs.

- [`check_canonical_phase_history.py`](/Users/armand/Development/RingRift/ai-service/scripts/check_canonical_phase_history.py)
  - Canonical move/phase history validator for replay databases.

- [`check_ts_python_replay_parity.py`](/Users/armand/Development/RingRift/ai-service/scripts/check_ts_python_replay_parity.py)
  - Main TypeScript ↔ Python replay parity harness.

- [`diff_state_bundle.py`](/Users/armand/Development/RingRift/ai-service/scripts/diff_state_bundle.py)
  - Focused state diff tool for one parity failure bundle.

- [`export_replay_dataset.py`](/Users/armand/Development/RingRift/ai-service/scripts/export_replay_dataset.py)
  - Preferred replay DB to NPZ export path.

- [`jsonl_to_npz.py`](/Users/armand/Development/RingRift/ai-service/scripts/jsonl_to_npz.py)
  - Converts policy-bearing JSONL into NPZ training artifacts.

### Training operations

- [`run_training_loop.py`](/Users/armand/Development/RingRift/ai-service/scripts/run_training_loop.py)
  - Higher-level training loop wrapper for config-driven runs.

- [`auto_promote.py`](/Users/armand/Development/RingRift/ai-service/scripts/auto_promote.py)
  - Promotion helper for candidate-to-best checkpoint flow.

- [`check_sync_health.py`](/Users/armand/Development/RingRift/ai-service/scripts/check_sync_health.py)
  - Sync-path sanity check for data/model distribution.

- [`cluster_health_cli.py`](/Users/armand/Development/RingRift/ai-service/scripts/cluster_health_cli.py)
  - Operator-facing health summary for cluster state.

- [`cleanup_selfplay_data.sh`](/Users/armand/Development/RingRift/ai-service/scripts/cleanup_selfplay_data.sh)
  - Selfplay data cleanup helper for reclaiming dead storage.

- [`db_health_check.py`](/Users/armand/Development/RingRift/ai-service/scripts/db_health_check.py)
  - Replay database integrity and health triage.

## Current Supported Flow

1. Deploy the role-based runtime with [`deploy_training_service.sh`](/Users/armand/Development/RingRift/ai-service/scripts/deploy_training_service.sh).
2. Trainers run [`minimal_alphazero_loop.py`](/Users/armand/Development/RingRift/ai-service/scripts/minimal_alphazero_loop.py).
3. Selfplay workers run [`policy_selfplay_worker.py`](/Users/armand/Development/RingRift/ai-service/scripts/policy_selfplay_worker.py).
4. Workers stage supplemental NPZ shards via [`ingest_policy_selfplay.py`](/Users/armand/Development/RingRift/ai-service/scripts/ingest_policy_selfplay.py).
5. Fleet state is checked with [`autonomy_fleet_check.py`](/Users/armand/Development/RingRift/ai-service/scripts/autonomy_fleet_check.py).

## Everything Else

Most remaining scripts fall into one of these categories:

- one-off incident tooling
- analysis and diagnosis helpers
- historical experiments
- provider-specific migration/deployment helpers

Do not assume those scripts are part of the supported path unless they are listed above or referenced by current docs under [`docs/architecture`](/Users/armand/Development/RingRift/ai-service/docs/architecture) or [`docs/runbooks`](/Users/armand/Development/RingRift/ai-service/docs/runbooks).
