# AI Service Scripts

This directory contains hundreds of scripts. Only a small subset is part of the supported operational surface.

If you are new to the codebase, start with the scripts below and treat most `analyze_*`, `benchmark_*`, and `debug_*` files as investigative tools rather than supported entrypoints.

## Essential Scripts

### Fleet runtime

- [`docs/operations/TRAINING_FLEET_RUNBOOK.md`](/docs/operations/TRAINING_FLEET_RUNBOOK.md)
  - Operator runbook for preflight, canary rollout, role-aware systemd deployment, health checks, reboot behavior, and rollback.

- [`docs/data/training_fleet_manifest.json`](/docs/data/training_fleet_manifest.json)
  - Checked-in role/config orientation manifest for the training fleet. Host inventory still comes from private runtime config.

- [`validate_training_fleet_docs.py`](/ai-service/scripts/validate_training_fleet_docs.py)
  - Read-only local preflight that cross-checks the fleet manifest, runbook, role file, canary deploy script, and systemd units before any SSH or deployment action.

- [`minimal_alphazero_loop.py`](/ai-service/scripts/minimal_alphazero_loop.py)
  - Canonical trainer loop for self-play, train, and evaluate on a single config.

- [`deploy_minimal_loops.sh`](/ai-service/scripts/deploy_minimal_loops.sh)
  - Supported canary-trainer deployment entrypoint for the minimal-loop fleet.
  - Runs a local minimal-loop preflight test slice before restarting trainer nodes.

- [`policy_selfplay_worker.py`](/ai-service/scripts/policy_selfplay_worker.py)
  - Dedicated policy-bearing Gumbel selfplay worker for role-based fleet nodes.

- [`ingest_policy_selfplay.py`](/ai-service/scripts/ingest_policy_selfplay.py)
  - Validates, converts, and stages policy selfplay JSONL into trainer supplemental NPZ shards.

- [`deploy_training_service.sh`](/ai-service/scripts/deploy_training_service.sh)
  - Role-aware systemd deployment for trainer, selfplay-worker, evaluator, and P2P services.

- [`autonomy_fleet_check.py`](/ai-service/scripts/autonomy_fleet_check.py)
  - Fleet health probe for the current role-based GH200 deployment.

- [`fleet_health_check.py`](/ai-service/scripts/fleet_health_check.py)
  - Broader fleet health diagnostics across hosts and services.

- [`p2p_orchestrator.py`](/ai-service/scripts/p2p_orchestrator.py)
  - P2P control plane for model sync, job coordination, and node health.

### Canonical data and parity

- [`generate_canonical_selfplay.py`](/ai-service/scripts/generate_canonical_selfplay.py)
  - Canonical selfplay generation with parity and history gating.

- [`run_canonical_selfplay_parity_gate.py`](/ai-service/scripts/run_canonical_selfplay_parity_gate.py)
  - Batch parity gate for canonical selfplay outputs.

- [`check_canonical_phase_history.py`](/ai-service/scripts/check_canonical_phase_history.py)
  - Canonical move/phase history validator for replay databases.

- [`check_ts_python_replay_parity.py`](/ai-service/scripts/check_ts_python_replay_parity.py)
  - Main TypeScript ↔ Python replay parity harness.

- [`diff_state_bundle.py`](/ai-service/scripts/diff_state_bundle.py)
  - Focused state diff tool for one parity failure bundle.

- [`export_replay_dataset.py`](/ai-service/scripts/export_replay_dataset.py)
  - Preferred replay DB to NPZ export path.

- [`jsonl_to_npz.py`](/ai-service/scripts/jsonl_to_npz.py)
  - Converts policy-bearing JSONL into NPZ training artifacts.

### Training operations

- [`run_training_loop.py`](/ai-service/scripts/run_training_loop.py)
  - Higher-level training loop wrapper for config-driven runs.

- [`auto_promote.py`](/ai-service/scripts/auto_promote.py)
  - Promotion helper for candidate-to-best checkpoint flow.

- [`check_sync_health.py`](/ai-service/scripts/check_sync_health.py)
  - Sync-path sanity check for data/model distribution.

- [`fleet_health_check.py`](/ai-service/scripts/fleet_health_check.py)
  - Operator-facing health summary for cluster state.

- [`cleanup_selfplay_data.sh`](/ai-service/scripts/cleanup_selfplay_data.sh)
  - Selfplay data cleanup helper for reclaiming dead storage.

- [`db_health_check.py`](/ai-service/scripts/db_health_check.py)
  - Replay database integrity and health triage.

## Current Supported Flow

1. Validate the checked-in fleet docs locally with [`validate_training_fleet_docs.py`](/ai-service/scripts/validate_training_fleet_docs.py).
2. For supported trainer canaries, deploy with [`deploy_minimal_loops.sh`](/ai-service/scripts/deploy_minimal_loops.sh).
3. That script preflights [`minimal_alphazero_loop.py`](/ai-service/scripts/minimal_alphazero_loop.py) locally before it restarts remote trainers.
4. Trainers write live state to `<work-dir>/progress.json` and durable history to `<work-dir>/metrics.jsonl`.
5. For the broader role-based fleet, use [`deploy_training_service.sh`](/ai-service/scripts/deploy_training_service.sh).
6. Use [`docs/operations/TRAINING_FLEET_RUNBOOK.md`](/docs/operations/TRAINING_FLEET_RUNBOOK.md) and [`docs/data/training_fleet_manifest.json`](/docs/data/training_fleet_manifest.json) to distinguish boot-persistent systemd services from the `nohup` minimal-loop canary supervisor.
7. Selfplay workers run [`policy_selfplay_worker.py`](/ai-service/scripts/policy_selfplay_worker.py).
8. Workers stage supplemental NPZ shards via [`ingest_policy_selfplay.py`](/ai-service/scripts/ingest_policy_selfplay.py).
9. Fleet state is checked with [`autonomy_fleet_check.py`](/ai-service/scripts/autonomy_fleet_check.py).

## Everything Else

Most remaining scripts fall into one of these categories:

- one-off incident tooling
- analysis and diagnosis helpers
- historical experiments
- provider-specific migration/deployment helpers

Archived scripts that are no longer part of the supported operational surface
now live under [`scripts/archive`](/ai-service/scripts/archive). If a command or helper only exists there, treat it as historical reference material unless a current runbook explicitly says otherwise.

Do not assume those scripts are part of the supported path unless they are listed above or referenced by current docs under [`docs/architecture`](/ai-service/docs/architecture) or [`docs/runbooks`](/ai-service/docs/runbooks).
